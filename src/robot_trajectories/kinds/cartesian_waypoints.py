from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..cartesian import poses_for, solve_points, tool_target_hold
from ..registry import LIMITS, kind
from ..segments import TOOL_TARGET, Piece
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator


@kind(use_cases=("tracking_eval", "stop_accuracy", "accuracy", "cycle_time", "demo"),
      needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"), unchecked=("collision_model",), catalog_id="cartesian_waypoints")
def cartesian_waypoints(space, points_or_poses, *,
                        waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5), orientation="hold",
                        dwell_s=0.2, back_to_start=False, tolerance_m=1e-4, label="cartesian waypoints", start=None):
    """The tool visits each of ``points_or_poses`` (xyz in metres, or 4x4 poses) and dwells there; between them the joints move
    in rest-to-rest legs under ``waypoint_generator`` (inverse kinematics at the points only, each seeded from the last): the Cartesian
    twin of ``joint_waypoints``. The tool does not move in straight lines between points (that is ``cartesian_path``).

    ``orientation``: "hold" (the start's) or "follow" (each pose's own). A point with no solution inside the limits is refused,
    naming it and why."""
    targets = poses_for(space, start, points_or_poses, orientation)
    if not targets:
        raise ValueError("cartesian_waypoints needs at least one point or pose")
    solutions = solve_points(space, start, targets, what=label, tolerance_m=tolerance_m)
    stops = list(zip(solutions, [target[:3, 3] for target in targets], strict=True))
    if back_to_start:
        stops.append((start, space.forward(start)[:3, 3]))
    pieces: list[k.Arrays] = []
    here = start
    for index, (q, point) in enumerate(stops, start=1):
        leg = waypoint_generator.leg(here, q, space.limits, space.rate_hz, label=f"{label}: leg {index}",
                                        dynamics=space.dynamics)
        target = np.full((len(leg.t), 3), np.nan)
        target[-1] = point  # WHY only the last row: between points the tool's path is not commanded
        pieces.append(Piece(leg.t, leg.q, leg.qd, leg.qdd, leg.label, channels={TOOL_TARGET: target}))
        if dwell_s > 0:
            rest = k.hold(q, dwell_s, space.rate_hz, label=f"{label}: dwell")
            pieces.append(Piece(rest.t, rest.q, rest.qd, rest.qdd, rest.label, channels=tool_target_hold(point, len(rest.t))))
        here = q
    return pieces
