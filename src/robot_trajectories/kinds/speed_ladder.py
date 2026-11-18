from __future__ import annotations

import math
import random

import numpy as np

from .. import kernels as k

from ..kind import Context, Kind, param
from ..time_law import TimeLaw

HELD_OUT_SPAN = (0.4, 0.9)  # a held-out move's |B - A| as a fraction of the repeat amplitude (pendulum_ladder)
HELD_OUT_MIN_DIFFERENCE_RAD = 0.05  # |A - A0| + |B - B0| from the repeat pair, so a held-out move is another input
HELD_OUT_ATTEMPTS = 2000
PRELUDE_SPEED_FRACTION = 0.5  # the runner's move to the origin (experiments.PRELUDE_SPEED_FRACTION)


class SpeedLadder(Kind):
    """Minimum-jerk A -> B -> A moves whose peak speed is a rising fraction of the joint's top speed, each rung repeated
    with the same input, plus held-out A/B pairs: tracking and stop accuracy as speed rises to the motor's limit (the
    pendulum campaign's frozen contract, pendulum_goal.py and pendulum_ladder.py).

    One A-to-B path under several time laws: ``ladder.time_laws`` is one ``TimeLaw.minimum_jerk(peak_speed_fraction=f)``
    per rung, and every rung plays the same A/B. Geometry as pendulum_ladder: the usable window is the room less a
    margin (5 deg or 5 % of the span, the larger) at both ends; A/B span it, A the end nearer the start; a held-out pair
    is drawn from a seed (round x 1000 + rung x 10 + k) spanning 0.4-0.9 of it. A rung whose peak the acceleration limit
    forbids is played as fast as the limit allows and the program notes say so (the ladder's ``clamped`` rows)."""

    catalog_id = "speed_ladder_a_to_b"
    family = "standard_test"
    starts_as = "path"
    use_cases = ("tracking_eval", "stop_accuracy", "speed_limits", "repeatability", "pid_tuning")
    needs = ("joint_limits", "velocity_limits", "acceleration_limits")

    joint: str | int | None = param(None, "joint", "the joint that moves (name or index; optional on a 1-DOF robot)")
    rung_fractions: tuple[float, ...] = param((0.2, 0.4, 0.6, 0.8, 0.9, 1.0), "1", "each rung's peak speed, a fraction of top speed")
    repeats: int = param(5, "count", "same-input repeats per rung: the repeat floor")
    held_out: int = param(2, "count", "other A/B per rung, scored only")
    shape: str = param("minimum_jerk", "text", "the time law's shape")
    dwell_s: float = param(1.0, "s", "the hold at B and back at A, where the settled stop is read")
    margin_rad: float = param(math.radians(5.0), "rad", "kept clear of each end of the room (or margin_fraction of the span, the larger)")
    margin_fraction: float = param(0.05, "1", "kept clear of each end, as a fraction of the span")
    round: int = param(0, "count", "the campaign round: it seeds the held-out pairs, so every round's are new")

    def check(self) -> None:
        if not self.rung_fractions or any(not 0 < fraction <= 1 for fraction in self.rung_fractions):
            raise ValueError(f"SpeedLadder.rung_fractions must lie in (0, 1], got {self.rung_fractions}")
        if self.repeats < 1 or self.held_out < 0:
            raise ValueError(f"SpeedLadder needs repeats >= 1 and held_out >= 0, got {self.repeats}, {self.held_out}")

    @property
    def time_laws(self) -> tuple[TimeLaw, ...]:
        """The ladder's timing: one time law per rung over the same A-to-B path."""
        return tuple(TimeLaw(self.shape, peak_speed_fraction=fraction) for fraction in self.rung_fractions)

    def window(self, context: Context) -> tuple[int, float, float, float, float]:
        """(joint, usable low, usable high, A, B) in rad for this robot."""
        joint = context.joint(self.joint)
        room = context.room_around(context.start)
        low, high = float(room.low[joint]), float(room.high[joint])
        margin = max(self.margin_rad, self.margin_fraction * (high - low))
        usable_low, usable_high = low + margin, high - margin
        if usable_high - usable_low < math.radians(3.0):
            raise ValueError(f"the usable window is {math.degrees(max(usable_high - usable_low, 0.0)):.1f}° (room "
                             f"{math.degrees(low):+.1f}..{math.degrees(high):+.1f}° less {math.degrees(margin):.1f}° at each end); "
                             "a ladder needs 3°: lower margin_rad / margin_fraction")
        rest = float(context.start[joint])
        a, b = (usable_low, usable_high) if abs(usable_low - rest) <= abs(usable_high - rest) else (usable_high, usable_low)
        return joint, usable_low, usable_high, a, b

    def held_out_pair(self, joint_name: str, seed: int, usable_low: float, usable_high: float, a0: float, b0: float,
                      rest: float) -> tuple[float, float]:
        """A seeded A/B inside the usable window, another input than the repeat pair (pendulum_ladder.held_out_pair)."""
        rng = random.Random(f"speed_ladder:{joint_name}:{seed}")
        amplitude = abs(b0 - a0)
        for _ in range(HELD_OUT_ATTEMPTS):
            span = rng.uniform(*HELD_OUT_SPAN) * amplitude
            low = rng.uniform(usable_low, usable_high - span)
            first, second = low, low + span
            a, b = (first, second) if abs(first - rest) <= abs(second - rest) else (second, first)
            if abs(a - a0) + abs(b - b0) >= HELD_OUT_MIN_DIFFERENCE_RAD:
                return a, b
        raise ValueError(f"no held-out A/B for seed {seed} in {HELD_OUT_ATTEMPTS} draws")

    def pieces(self, context: Context) -> list[k.Arrays]:
        joint, usable_low, usable_high, a0, b0 = self.window(context)
        name = context.joint_names[joint]
        rest = float(context.start[joint])

        def at(value: float) -> np.ndarray:
            q = context.start.copy()
            q[joint] = value
            return q

        prelude = TimeLaw(self.shape, peak_speed_fraction=PRELUDE_SPEED_FRACTION)
        pieces = [prelude.leg(context.start, at(a0), context.bounds, context.rate_hz, label=f"{self.display_label}: to A")]
        here = at(a0)
        clamped = []
        for rung, law in enumerate(self.time_laws):
            percent = f"{100 * law.peak_speed_fraction:.0f} %"
            timing = law.leg_timing(at(a0), at(b0), context.bounds)
            if timing.clamped:
                clamped.append(f"{percent} peaks at {100 * timing.peak_speed_fraction:.0f} % ({timing.bound}-bound)")
            pairs = [(a0, b0, f"repeat {index + 1}/{self.repeats}") for index in range(self.repeats)]
            for index in range(self.held_out):
                seed = 1000 * self.round + 10 * rung + index
                a, b = self.held_out_pair(name, seed, usable_low, usable_high, a0, b0, rest)
                pairs.append((a, b, f"held out {index + 1}/{self.held_out} (seed {seed})"))
            for a, b, what in pairs:
                label = f"{self.display_label}: {percent} {what}"
                if not np.isclose(here[joint], a):
                    pieces.append(prelude.leg(here, at(a), context.bounds, context.rate_hz, label=f"{label}: to A"))
                # the same path, this rung's time law: A -> B (dwell) -> A (dwell)
                pieces += [law.leg(at(a), at(b), context.bounds, context.rate_hz, label=f"{label}: A to B"),
                           k.hold(at(b), self.dwell_s, context.rate_hz, label=f"{label}: dwell at B"),
                           law.leg(at(b), at(a), context.bounds, context.rate_hz, label=f"{label}: B to A"),
                           k.hold(at(a), self.dwell_s, context.rate_hz, label=f"{label}: dwell at A")]
                here = at(a)
        pieces.append(prelude.leg(here, context.start, context.bounds, context.rate_hz, label=f"{self.display_label}: back to start"))
        context.note(f"{self.display_label}: A/B {math.degrees(a0):+.1f}° / {math.degrees(b0):+.1f}° on {name}; "
                     + ("; ".join(clamped) + ": a limit binds before the rung's speed" if clamped else "every rung reaches its speed"))
        return pieces
