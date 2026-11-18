from __future__ import annotations

import math

from ..registry import kind
from ..segments import annotate, chain
from .home import home
from .workspace_sweep import workspace_sweep


@kind(use_cases=("workspace_coverage", "accuracy", "kinematic_calibration"), family="coverage", starts_as="path",
      dimensions=("2D", "3D", "6D"), better_with={"reach_map": "a box grid, every configuration in it assumed reachable"},
      unchecked=("collision_model",), higher_order=True)
def workspace_survey(space, *, grid_step=math.radians(12.0), joints=None, points_per_joint=3, speed=0.8, dwell_s=0.05, start=None):
    """Where can it go, and how accurately? Every reachable configuration on a ``grid_step`` grid, neighbour moves only (a
    ``points_per_joint`` box grid without a reach map), then home: coverage, accuracy, calibration poses."""
    sweep = annotate(workspace_sweep(space, grid_step=grid_step, joints=joints, points_per_joint=points_per_joint, speed=speed,
                                     dwell_s=dwell_s, start=start),
                     f"every reachable configuration on a {math.degrees(grid_step):g}° grid, neighbour moves only (a "
                     f"{points_per_joint}-per-joint box grid without a reach map): coverage, accuracy, calibration poses")
    back = annotate(home(space, start=sweep), "back home", wait_after=True)
    return chain(sweep, back)
