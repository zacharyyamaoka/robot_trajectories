from __future__ import annotations

import numpy as np

from .. import kernels as k

from ..kind import Context, Kind, param
from ..time_law import TimeLaw


class Move(Kind):
    """Rest-to-rest moves through waypoints, each leg timed by a TimeLaw (minimum-jerk by default: x = x0 + d (10 s^3 -
    15 s^4 + 6 s^5), peaking at 1.875 d / T), with a dwell at each, then back to the start: the house leg of every
    waypoints tape (the rack runner's ``q:move_s:dwell_s;...``).

    A path: the waypoints are the content, ``time_law`` is the timing. Waypoints are fractions of the room around the
    start (as the runner's are relative to its origin), so one record runs on any robot."""

    catalog_id = "minimum_jerk_move"
    family = "point_to_point"
    starts_as = "path"
    use_cases = ("tracking_eval", "stop_accuracy", "pid_tuning", "demo")
    needs = ("joint_limits", "velocity_limits", "acceleration_limits")

    joint: str | int | None = param(None, "joint", "the joint that moves (default: every joint)")
    targets_fraction: tuple = param((0.5, -0.5), "1", "each waypoint: a fraction of the room (one per moving joint, or one for all)")
    time_law: TimeLaw = param(TimeLaw.minimum_jerk(move_s=1.0), "time law", "how each leg is timed (the catalog's move_s 1 s)")
    dwell_s: float = param(0.3, "s", "the hold at each waypoint")
    back_to_start: bool = param(True, "bool", "end with a leg back to the start (the runner's waypoints always do)")

    def check(self) -> None:
        if not self.targets_fraction:
            raise ValueError("Move needs at least one waypoint in targets_fraction")
        for target in self.targets_fraction:
            values = target if isinstance(target, tuple) else (target,)
            if any(abs(value) > 1.0 for value in values):
                raise ValueError(f"Move.targets_fraction must lie in [-1, 1], got {target}")

    def waypoints(self, context: Context) -> np.ndarray:
        """The path: absolute joint configurations (M, J) for this robot."""
        joints = context.joints(self.joint)
        room = context.room_around(context.start)
        half = np.minimum(context.start - room.low, room.high - context.start)
        rows = []
        for target in self.targets_fraction:
            values = np.broadcast_to(np.asarray(target, float), (len(joints),))
            row = context.start.copy()
            row[joints] += values * half[joints]
            rows.append(row)
        if self.back_to_start:
            rows.append(context.start.copy())
        return np.array(rows)

    def pieces(self, context: Context) -> list[k.Arrays]:
        pieces: list[k.Arrays] = []
        here = context.start
        for index, target in enumerate(self.waypoints(context), start=1):
            pieces.append(self.time_law.leg(here, target, context.bounds, context.rate_hz, label=f"{self.display_label}: leg {index}"))
            if self.dwell_s > 0:
                pieces.append(k.hold(target, self.dwell_s, context.rate_hz, label=f"{self.display_label}: dwell"))
            here = target
        return pieces
