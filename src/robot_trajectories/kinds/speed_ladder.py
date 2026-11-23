from __future__ import annotations

import math
import random

import numpy as np

from .. import kernels as k
from ..registry import kind
from ..segments import Motion
from ..time_parameterization import TimeParameterization

RUNG_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
HELD_OUT_SPAN = (0.4, 0.9)  # a held-out move's |B - A| as a fraction of the repeat amplitude (pendulum_ladder)
HELD_OUT_MIN_DIFFERENCE_RAD = 0.05  # |A - A0| + |B - B0| from the repeat pair, so a held-out move is another input
HELD_OUT_ATTEMPTS = 2000
PRELUDE_SPEED_FRACTION = 0.5  # the runner's move to the origin (experiments.PRELUDE_SPEED_FRACTION)


def time_parameterizations(rung_fractions=RUNG_FRACTIONS, shape: str = "minimum_jerk") -> tuple[TimeParameterization, ...]:
    """The ladder's timing: one time parameterization per rung over the same A-to-B path."""
    return tuple(TimeParameterization(shape, peak_speed_fraction=fraction) for fraction in rung_fractions)


time_laws = time_parameterizations  # deprecated alias, silent (the rename of 2026-11-23; time_law.py says why)


def held_out_pair(joint_name: str, seed: int, usable_low: float, usable_high: float, a0: float, b0: float,
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


@kind(use_cases=("tracking_eval", "stop_accuracy", "speed_limits", "repeatability", "pid_tuning"), catalog_id="speed_ladder_a_to_b")
def speed_ladder(space, joint=None, *, rung_fractions=RUNG_FRACTIONS, repeats=5, held_out=2, shape="minimum_jerk", dwell_s=1.0,
                 margin_rad=math.radians(5.0), margin_fraction=0.05, campaign_round=0, label="speed ladder", start=None):
    """Rest-to-rest A -> B -> A moves whose peak speed is a rising fraction of the joint's top speed, each rung repeated with the
    same input (``repeats``), plus held-out A/B pairs (``held_out``, scored only): tracking and stop accuracy as speed rises to
    the motor's limit (the pendulum campaign's frozen contract, pendulum_goal.py and pendulum_ladder.py).

    One A-to-B path under several time parameterizations: ``time_parameterizations(rung_fractions, shape)``, and every rung plays the same A/B.
    Geometry as pendulum_ladder: the usable window is the room less a margin (``margin_rad`` or ``margin_fraction`` of the
    span, the larger) at both ends; A/B span it, A the end nearer the start; a held-out pair is drawn from a seed
    (campaign_round x 1000 + rung x 10 + k) spanning 0.4-0.9 of it. A rung whose peak the acceleration limit forbids is
    played as fast as the limit allows and the notes say so (the ladder's ``clamped`` rows)."""
    if not rung_fractions or any(not 0 < fraction <= 1 for fraction in rung_fractions):
        raise ValueError(f"speed_ladder.rung_fractions must lie in (0, 1], got {rung_fractions}")
    if repeats < 1 or held_out < 0:
        raise ValueError(f"speed_ladder needs repeats >= 1 and held_out >= 0, got {repeats}, {held_out}")
    joint = space.joint_index(joint)
    room = space.room(start)
    low, high = float(room.low[joint]), float(room.high[joint])
    margin = max(margin_rad, margin_fraction * (high - low))
    usable_low, usable_high = low + margin, high - margin
    if usable_high - usable_low < math.radians(3.0):
        raise ValueError(f"the usable window is {math.degrees(max(usable_high - usable_low, 0.0)):.1f}° (room "
                         f"{math.degrees(low):+.1f}..{math.degrees(high):+.1f}° less {math.degrees(margin):.1f}° at each end); "
                         "a ladder needs 3°: lower margin_rad / margin_fraction")
    rest = float(start[joint])
    a0, b0 = (usable_low, usable_high) if abs(usable_low - rest) <= abs(usable_high - rest) else (usable_high, usable_low)
    name = space.joint_names[joint]

    def at(value: float) -> np.ndarray:
        q = start.copy()
        q[joint] = value
        return q

    limits, rate = space.limits, space.rate_hz
    prelude = TimeParameterization(shape, peak_speed_fraction=PRELUDE_SPEED_FRACTION)
    pieces = [prelude.leg(start, at(a0), limits, rate, label=f"{label}: to A", dynamics=space.dynamics)]
    here = at(a0)
    clamped = []
    for rung, law in enumerate(time_parameterizations(rung_fractions, shape)):
        percent = f"{100 * law.peak_speed_fraction:.0f} %"
        timing = law.leg_timing(at(a0), at(b0), limits)
        if timing.clamped:
            clamped.append(f"{percent} peaks at {100 * timing.peak_speed_fraction:.0f} % ({timing.bound}-bound)")
        pairs = [(a0, b0, f"repeat {index + 1}/{repeats}") for index in range(repeats)]
        for index in range(held_out):
            seed = 1000 * campaign_round + 10 * rung + index
            a, b = held_out_pair(name, seed, usable_low, usable_high, a0, b0, rest)
            pairs.append((a, b, f"held out {index + 1}/{held_out} (seed {seed})"))
        for a, b, what in pairs:
            band = f"{label}: {percent} {what}"
            if not np.isclose(here[joint], a):
                pieces.append(prelude.leg(here, at(a), limits, rate, label=f"{band}: to A", dynamics=space.dynamics))
            # the same path, this rung's time parameterization: A -> B (dwell) -> A (dwell)
            pieces += [law.leg(at(a), at(b), limits, rate, label=f"{band}: A to B", dynamics=space.dynamics),
                       k.hold(at(b), dwell_s, rate, label=f"{band}: dwell at B"),
                       law.leg(at(b), at(a), limits, rate, label=f"{band}: B to A", dynamics=space.dynamics),
                       k.hold(at(a), dwell_s, rate, label=f"{band}: dwell at A")]
            here = at(a)
    pieces.append(prelude.leg(here, start, limits, rate, label=f"{label}: back to start", dynamics=space.dynamics))
    note = (f"{label}: A/B {math.degrees(a0):+.1f}° / {math.degrees(b0):+.1f}° on {name}; "
            + ("; ".join(clamped) + ": a limit binds before the rung's speed" if clamped else "every rung reaches its speed"))
    return Motion(pieces, [note])
