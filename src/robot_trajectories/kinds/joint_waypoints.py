from __future__ import annotations

import math

import numpy as np

from ..registry import kind
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator, legs_through


@kind(use_cases=("tracking_eval", "stop_accuracy", "cycle_time", "demo"), catalog_id="joint_waypoints")
def joint_waypoints(space, waypoints, *, waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5),
                    dwell_s=0.2, back_to_start=False, label="joint waypoints", start=None):
    """Rest-to-rest moves through absolute joint configurations (``waypoints``, M x J rad; a flat list on a 1-DOF robot), each
    leg timed by ``waypoint_generator``, dwelling ``dwell_s`` at each: multi-waypoint motion in joint space (the classic PTP sequence).

    ``back_to_start`` ends with a leg back to where it began. Every waypoint must lie inside the position limits; the first
    one outside is refused by name. ``move`` takes fractions of the room instead; ``cartesian_waypoints`` takes tool points."""
    rows = np.asarray(waypoints, float)
    rows = rows.reshape(-1, space.dof) if rows.size % space.dof == 0 else rows
    if rows.ndim != 2 or rows.shape[1] != space.dof or len(rows) == 0:
        raise ValueError(f"waypoints must be M x {space.dof} joint configurations ({space.joint_names}), got shape {np.shape(waypoints)}")
    bounds = space.bounds
    for index, row in enumerate(rows):
        outside = np.flatnonzero((row < bounds.low - 1e-12) | (row > bounds.high + 1e-12))
        if len(outside):
            joint = int(outside[0])
            raise ValueError(f"waypoint {index} puts {space.joint_names[joint]} at {math.degrees(row[joint]):.2f}°, outside its limits "
                             f"{math.degrees(bounds.low[joint]):.2f}..{math.degrees(bounds.high[joint]):.2f}°")
    targets = [*rows, start] if back_to_start else list(rows)
    return legs_through(start, targets, waypoint_generator, space.limits, space.rate_hz, dwell_s=dwell_s, label=label,
                        dynamics=space.dynamics)
