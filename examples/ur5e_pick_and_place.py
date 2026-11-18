"""The U pick-and-place on a UR5e: two tool points in, one joint trajectory out, with the kinematics passed in.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/examples/ur5e_pick_and_place.py

WHY the robot comes from tests/fixtures/robots.py: the package holds no robot. The UR5e there is a stand-in shaped like
bam_reach's Kinematics (DH forward kinematics, damped-least-squares inverse); a real space passes its own.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from fixtures import robots  # noqa: E402

from robot_trajectories import JointSpace, pick_and_place, split, summary  # noqa: E402

space = (JointSpace.from_description(robots.UR5E_DESCRIPTION, max_acceleration=6.0)   # the URDF has no acceleration: state one
         .within(robots.UR5E_DEMO_BOX)                                                # a box above the table
         .with_kinematics(robots.UR5eKinematics()))                                  # FK / IK, injected

tool = space.forward(space.home)[:3, 3]                                              # where the tool is at home (m)
cycle = pick_and_place(space, pick=tool + (0.08, -0.08, -0.12), place=tool + (-0.08, 0.10, -0.12), lift_m=0.08)

print(summary(cycle))
reached = space.tool_points(cycle)
print(f"the tool at the two dwells: {np.round(1000 * reached[np.searchsorted(cycle.t, split(cycle)[2].header.stamp + 0.1)], 1)} mm, "
      f"{np.round(1000 * reached[np.searchsorted(cycle.t, split(cycle)[6].header.stamp + 0.1)], 1)} mm")
