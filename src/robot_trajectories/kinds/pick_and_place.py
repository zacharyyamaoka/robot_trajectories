from __future__ import annotations

import numpy as np

from ..registry import LIMITS, kind
from ..segments import annotate, chain
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator
from .cartesian_path import cartesian_path
from .cartesian_waypoints import cartesian_waypoints
from .hold import hold


@kind(use_cases=("cycle_time", "tracking_eval", "stop_accuracy", "demo", "conveyor_pick_and_place"),
      needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"), unchecked=("collision_model",), catalog_id="pick_and_place_cycle",
      higher_order=True)
def pick_and_place(space, pick, place, *, lift_m=0.05, up=(0.0, 0.0, 1.0),
                   waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.3),
                   approach_waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5), dwell_s=0.5,
                   eef_step_m=0.001, tolerance_m=1e-4, start=None):
    """The U between two tool points (xyz, metres): approach above the first (a joint move), down to it in a straight line,
    dwell, up, over to above the second, down, dwell, up. ``lift_m`` along ``up`` (a direction in the base frame; on a
    planar robot its plane's up). The tool keeps its start orientation throughout.

    Task-agnostic: two points and a lift. What happens at the dwells is the caller's (each dwell segment says wait_after, so
    ``split`` and run one at a time). A higher-order kind: cartesian_waypoints, cartesian_path and hold, chained."""
    direction = np.asarray(up, float)
    if np.linalg.norm(direction) < 1e-12:
        raise ValueError("up must be a nonzero direction")
    direction = direction / np.linalg.norm(direction)
    rotation = space.forward(start)[:3, :3]

    def pose(point: np.ndarray) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, point
        return matrix

    first, second = np.asarray(pick, float), np.asarray(place, float)
    above_first, above_second = first + lift_m * direction, second + lift_m * direction
    line = {"waypoint_generator": waypoint_generator, "orientation": "follow", "eef_step_m": eef_step_m,
            "tolerance_m": tolerance_m}
    approach = cartesian_waypoints(space, [pose(above_first)], waypoint_generator=approach_waypoint_generator,
                                   orientation="follow", dwell_s=0.0, tolerance_m=tolerance_m, label="approach", start=start)
    down_1 = cartesian_path(space, [pose(first)], label="down to the first point", start=approach, **line)
    rest_1 = annotate(hold(space, dwell_s, label="dwell at the first point", start=down_1), "at rest on the first point", wait_after=True)
    up_1 = cartesian_path(space, [pose(above_first)], label="up from the first point", start=rest_1, **line)
    over = cartesian_path(space, [pose(above_second)], label="over", start=up_1, **line)
    down_2 = cartesian_path(space, [pose(second)], label="down to the second point", start=over, **line)
    rest_2 = annotate(hold(space, dwell_s, label="dwell at the second point", start=down_2), "at rest on the second point",
                      wait_after=True)
    up_2 = cartesian_path(space, [pose(above_second)], label="up from the second point", start=rest_2, **line)
    return chain(approach, down_1, rest_1, up_1, over, down_2, rest_2, up_2)
