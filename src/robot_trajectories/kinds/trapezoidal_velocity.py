from __future__ import annotations

import math

import numpy as np

from .. import kernels as k
from ..registry import kind
from ..segments import Motion
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator


@kind(use_cases=("tracking_eval", "cycle_time"), catalog_id="trapezoidal_velocity")
def trapezoidal_velocity(space, target=None, *, velocity_fraction=1.0, acceleration_fraction=1.0, dwell_s=0.0, back_to_start=False,
                         label="trapezoidal velocity", start=None):
    """One synchronised joint move to ``target`` (absolute joint positions, rad) under a trapezoidal velocity profile: accelerate,
    cruise, decelerate (Pilz PTP's profile; a triangle when the distance is too short to cruise). Every joint arrives together:
    the slowest one sets the time, the others are scaled to it, so the joint-space line is straight.

    ``velocity_fraction`` and ``acceleration_fraction`` scale the space's velocity and acceleration limits (Pilz's velocity and
    acceleration scaling); the move is as fast as they allow. ``target=None`` moves half way to the top of each joint's room.
    ``dwell_s`` holds at the target, ``back_to_start`` returns. The ramps are a quarter of the leg each (``WaypointTrajectoryGenerator.trapezoid``);
    the acceleration steps at their ends, so the jerk is unbounded: minimum-jerk (``move``, ``joint_waypoints``) is the smooth one."""
    for name, value in (("velocity_fraction", velocity_fraction), ("acceleration_fraction", acceleration_fraction)):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"trapezoidal_velocity.{name} must lie in (0, 1], got {value}")
    if target is None:
        goal = start + 0.5 * (space.room(start).high - start)
    else:
        goal = np.asarray(target, float).reshape(-1)
        if goal.shape != (space.dof,):
            raise ValueError(f"target needs {space.dof} joint positions {space.joint_names}, got {goal.size}")
        bounds = space.bounds
        outside = np.flatnonzero((goal < bounds.low - 1e-12) | (goal > bounds.high + 1e-12))
        if len(outside):
            joint = int(outside[0])
            raise ValueError(f"target puts {space.joint_names[joint]} at {math.degrees(goal[joint]):.2f}°, outside its limits "
                             f"{math.degrees(bounds.low[joint]):.2f}..{math.degrees(bounds.high[joint]):.2f}°")
    bounds = space.bounds
    # WHY the scaled limits go to the law at full speed: the law then runs each joint at its scaled velocity or acceleration
    # limit, whichever binds first (Pilz PTP's scaling), and the guard still holds the output to the space's own limits
    scaled = k.Bounds(bounds.joint_names, bounds.low, bounds.high, bounds.max_velocity * velocity_fraction,
                      bounds.max_acceleration * acceleration_fraction)
    law = WaypointTrajectoryGenerator.trapezoid(peak_speed_fraction=1.0)
    pieces = [law.leg(start, goal, scaled, space.rate_hz, label=f"{label}: leg")]
    timing = law.leg_timing(start, goal, scaled)
    if dwell_s > 0:
        pieces.append(k.hold(goal, dwell_s, space.rate_hz, label=f"{label}: dwell"))
    if back_to_start:
        pieces.append(law.leg(goal, start, scaled, space.rate_hz, label=f"{label}: leg back"))
    note = (f"{label}: {timing.duration_s:.3f} s, {timing.bound}-bound, the slowest joint peaks at "
            f"{100 * timing.peak_speed_fraction:.0f} % of its (scaled) top speed")
    return Motion(pieces, [note])
