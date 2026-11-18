from __future__ import annotations

import math

from .. import kernels as k

from ..kind import Context, Kind, param


class WorkspaceSweep(Kind):
    """Visit a grid of configurations: with a reach map, every reachable one, moving only between grid neighbours; without
    one, a serpentine grid over the box (V4's fallback). Coverage, accuracy, calibration poses."""

    catalog_id = "joint_grid_snake"
    family = "coverage"
    starts_as = "path"
    use_cases = ("workspace_coverage", "accuracy", "kinematic_calibration")
    needs = ("joint_limits", "velocity_limits", "acceleration_limits", "reach_map")
    without = {"reach_map": "a serpentine grid over the box, every configuration in it assumed reachable"}
    unchecked = ("jerk_limits", "collision_model")  # the catalog's wrist-coverage source times its legs with Ruckig

    grid_step: float = param(math.radians(12.0), "rad", "with a reach map: the visited grid (a whole multiple of the map's step)")
    joints: tuple[str, ...] | None = param(None, "joints", "without a reach map: the joints the grid spans (default every joint)")
    points_per_joint: int = param(3, "count", "without a reach map: grid points along each spanned joint")
    margin: float = param(0.05, "1", "without a reach map: fraction of each joint's range kept clear at both ends")
    speed: float = param(0.8, "1", "fraction of the velocity and acceleration limits a leg may use")
    dwell_s: float = param(0.05, "s", "the hold at each grid point")

    def pieces(self, context: Context) -> list[k.Arrays]:
        if context.reach is not None:
            grid = context.reach.reachable_on_grid(self.grid_step)
            tour = k.reachable_tour(grid, self.grid_step, context.start)  # neighbour moves only: never across a hole
            return k.waypoint_chain(tour, context.bounds, context.rate_hz, start=context.start, speed=self.speed,
                                    dwell_s=self.dwell_s, label=self.display_label)
        grid = k.serpentine_grid(context.bounds, self.points_per_joint, joints=context.joints(self.joints), margin=self.margin,
                                 rest=context.start)
        return k.waypoint_chain(grid, context.bounds, context.rate_hz, start=context.start, speed=self.speed,
                                dwell_s=self.dwell_s, label=self.display_label)
