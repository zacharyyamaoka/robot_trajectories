from __future__ import annotations

import math

from .. import kernels as k
from ..registry import kind


@kind(use_cases=("workspace_coverage", "accuracy", "kinematic_calibration"),
      better_with={"reach_map": "a serpentine grid over the box, every configuration in it assumed reachable"},
      unchecked=("jerk_limits", "collision_model"), catalog_id="joint_grid_snake")
def workspace_sweep(space, *, grid_step=math.radians(12.0), joints=None, points_per_joint=3, margin=0.05, speed=0.8, dwell_s=0.05,
                    label="workspace sweep", start=None):
    """Visit a grid of configurations: with a reach map, every reachable one on a ``grid_step`` grid (rad, a whole multiple of
    the map's step), moving only between grid neighbours so no leg crosses a hole; without one, a serpentine grid of
    ``points_per_joint`` over ``joints`` inside the box shrunk by ``margin``. Coverage, accuracy, calibration poses.

    ``speed``: the fraction of the velocity and acceleration limits a leg may use."""
    if space.reach is not None:
        grid = space.reach.reachable_on_grid(grid_step)
        tour = k.reachable_tour(grid, grid_step, start)  # neighbour moves only: never across a hole
        return k.waypoint_chain(tour, space.bounds, space.rate_hz, start=start, speed=speed, dwell_s=dwell_s, label=label)
    grid = k.serpentine_grid(space.bounds, points_per_joint, joints=space.joint_indices(joints), margin=margin, rest=start)
    return k.waypoint_chain(grid, space.bounds, space.rate_hz, start=start, speed=speed, dwell_s=dwell_s, label=label)
