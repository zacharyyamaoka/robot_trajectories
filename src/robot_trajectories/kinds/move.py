from __future__ import annotations

import numpy as np

from ..registry import kind
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator, legs_through


@kind(use_cases=("tracking_eval", "stop_accuracy", "pid_tuning", "demo"), catalog_id="minimum_jerk_move")
def move(space, joint=None, *, targets_fraction=(0.5, -0.5), waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(move_s=1.0),
         dwell_s=0.3, back_to_start=True, label="move", start=None):
    """Rest-to-rest moves through waypoints given as fractions of the room around the start, each leg timed by ``waypoint_generator``
    (minimum-jerk by default: x = x0 + d (10 s^3 - 15 s^4 + 6 s^5), peaking at 1.875 d / T), with a dwell at each, then back
    to the start: the house leg of every waypoints tape (the rack runner's ``q:move_s:dwell_s;...``).

    ``joint``: the joint that moves (default every joint). Each entry of ``targets_fraction`` is one waypoint: one fraction
    for every moving joint, or one per moving joint. Fractions of the room (as the runner's are relative to its origin), so
    one call runs on any robot; ``joint_waypoints`` takes absolute configurations."""
    for target in targets_fraction:
        values = target if isinstance(target, tuple) else (target,)
        if any(abs(value) > 1.0 for value in values):
            raise ValueError(f"move.targets_fraction must lie in [-1, 1], got {target}")
    if not targets_fraction:
        raise ValueError("move needs at least one waypoint in targets_fraction")
    joints = space.joint_indices(joint)
    room = space.room(start)
    half = np.minimum(start - room.low, room.high - start)
    rows = []
    for target in targets_fraction:
        values = np.broadcast_to(np.asarray(target, float), (len(joints),))
        row = start.copy()
        row[joints] += values * half[joints]
        rows.append(row)
    if back_to_start:
        rows.append(start.copy())
    return legs_through(start, rows, waypoint_generator, space.limits, space.rate_hz, dwell_s=dwell_s, label=label,
                        dynamics=space.dynamics)
