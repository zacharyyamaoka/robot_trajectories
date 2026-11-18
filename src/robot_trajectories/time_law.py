"""TimeLaw: how a path is timed. ``Path x time law -> JointTrajectory`` (Zach's refined spec, 2026-11-18).

A path kind (a move, the speed ladder, the deployment classes) says WHERE the joints go; its ``time_law`` says how each
leg of it is timed. The speed ladder is then one A-to-B path under six time laws:

    TimeLaw.minimum_jerk(peak_speed_fraction=0.6)   # each leg peaks at 60 % of top speed (no joint above 60 % of its own)
    TimeLaw.minimum_jerk(move_s=1.0)                # each leg takes 1 s (longer if a limit needs it)

WHY only minimum-jerk: it is the house leg (the rack runner's waypoints, PID tuna's tapes, the pendulum campaign). The
catalog's other time laws (trapezoid, S-curve via Ruckig, time-optimal via TOPP-RA) are refused by name until built.

Inside every limit by construction: a leg never runs faster than its law asks, and never faster than the velocity AND
acceleration limits allow. When the acceleration limit binds, the leg peaks lower than asked and ``leg_timing`` says so
(as the pendulum ladder's index marks a ``clamped`` rung).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import kernels as k

SHAPES = ("minimum_jerk",)
NOT_YET = ("trapezoidal_velocity", "jerk_limited_scurve", "polynomial_move", "time_optimal_path")  # catalog ids


@dataclass(frozen=True)
class LegTiming:
    """How one leg was timed: its duration, what bounded it, and the peak speed it reaches."""

    duration_s: float
    bound: str  # "speed" (the law's own number), "move_s", "acceleration", "velocity" or "min_move_s"
    peak_speed_fraction: float  # the worst joint's peak speed over its own top speed, as generated

    @property
    def clamped(self) -> bool:
        return self.bound in ("acceleration", "velocity", "min_move_s")


@dataclass(frozen=True)
class TimeLaw:
    """The shape of each leg's progress s(t), and either its peak speed or its duration."""

    shape: str = "minimum_jerk"
    peak_speed_fraction: float | None = None  # no joint's peak passes this fraction of its own top speed
    move_s: float | None = None  # a fixed time per leg instead
    min_move_s: float = 0.1  # WHY 0.1 s: ten 100 Hz plan ticks, the ladder's MIN_MOVE_S; a shorter leg is not smooth on the rig

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"time law {self.shape!r} is not built yet; built: {SHAPES} (the catalog's {NOT_YET} come later)")
        if (self.peak_speed_fraction is None) == (self.move_s is None):
            raise ValueError("a time law takes peak_speed_fraction= or move_s=, exactly one")
        if self.peak_speed_fraction is not None and not 0.0 < self.peak_speed_fraction <= 1.0:
            raise ValueError(f"peak_speed_fraction must lie in (0, 1], got {self.peak_speed_fraction}")
        if self.move_s is not None and self.move_s <= 0.0:
            raise ValueError(f"move_s must be positive, got {self.move_s}")

    @classmethod
    def minimum_jerk(cls, *, peak_speed_fraction: float | None = None, move_s: float | None = None,
                     min_move_s: float = 0.1) -> TimeLaw:
        """Minimum-jerk legs x = x0 + d (10 s^3 - 15 s^4 + 6 s^5), peaking at 1.875 d / T."""
        return cls("minimum_jerk", peak_speed_fraction=peak_speed_fraction, move_s=move_s, min_move_s=min_move_s)

    # ------------------------------------------------------------------ timing a leg

    def leg_timing(self, q0: np.ndarray, q1: np.ndarray, bounds: k.Bounds) -> LegTiming:
        """The leg's duration: the law's, raised until every joint is inside its velocity and acceleration limits."""
        distance = np.abs(np.asarray(q1, float) - np.asarray(q0, float))
        if not np.any(distance > 1e-12):
            return LegTiming(self.min_move_s, "min_move_s", 0.0)
        by_velocity = float(np.max(k.MIN_JERK_PEAK_VELOCITY * distance / bounds.max_velocity))
        by_acceleration = float(np.max(np.sqrt(k.MIN_JERK_PEAK_ACCELERATION * distance / bounds.max_acceleration)))
        if self.peak_speed_fraction is not None:
            # WHY per joint: "60 % of top speed" on a multi-joint leg means no joint passes 60 % of its own top speed.
            wanted, why = float(np.max(k.MIN_JERK_PEAK_VELOCITY * distance / (self.peak_speed_fraction * bounds.max_velocity))), "speed"
        else:
            wanted, why = float(self.move_s), "move_s"
        # WHY the law wins ties: a leg bounded exactly by what was asked is not "clamped"
        candidates = [(wanted, why), (by_velocity, "velocity"), (by_acceleration, "acceleration"), (self.min_move_s, "min_move_s")]
        duration, bound = max(candidates, key=lambda item: (item[0], item[1] == why))
        if duration <= wanted * (1 + 1e-12):
            bound = why
        # WHY rounded up to the ms (pendulum_ladder.move_time): the peak never exceeds what was asked
        duration = math.ceil(duration * 1000.0 - 1e-6) / 1000.0
        peak = float(np.max(k.MIN_JERK_PEAK_VELOCITY * distance / duration / bounds.max_velocity))
        return LegTiming(duration, bound, peak)

    def leg(self, q0: np.ndarray, q1: np.ndarray, bounds: k.Bounds, rate_hz: float, *, label: str = "move") -> k.Arrays:
        """One rest-to-rest leg q0 -> q1 under this law (the shared kernels' minimum-jerk leg)."""
        timing = self.leg_timing(q0, q1, bounds)
        return k.min_jerk(q0, q1, bounds, rate_hz, duration_s=timing.duration_s, label=label)

    # ------------------------------------------------------------------ as configuration

    def to_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"time_law": self.shape}
        if self.peak_speed_fraction is not None:
            config["peak_speed_fraction"] = self.peak_speed_fraction
        if self.move_s is not None:
            config["move_s"] = self.move_s
        config["min_move_s"] = self.min_move_s
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TimeLaw:
        values = dict(config)
        return cls(values.pop("time_law"), **values)

    def __repr__(self) -> str:
        parts = [f"peak_speed_fraction={self.peak_speed_fraction!r}" if self.peak_speed_fraction is not None else f"move_s={self.move_s!r}"]
        if self.min_move_s != 0.1:
            parts.append(f"min_move_s={self.min_move_s!r}")
        return f"TimeLaw.{self.shape}({', '.join(parts)})"
