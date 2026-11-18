"""Lane F's fixtures: the two robots with kinematics, targets moving at constant velocity, and the meet_moving_point kind.

- ``ur5e()`` / ``fivebar()``: the spaces tests/test_cartesian.py uses (the UR5e's demo box with its DH kinematics; the five-bar's
  limits with its closed-form tool point), and ``FIVEBAR_START``, inside its reach away from the elbow singularity.
- ``linear_target(...)``: a ``LinearMotion`` as the mapping a lane-F kind takes (and its recipe keeps), ``dataclasses.asdict``.
- ``meet_moving_point``: the kind ``robot_trajectories.meet_moving_point`` (catalog row ``meet_moving_point``), re-exported
  so the tests that chained ``track_and_release`` after this module's stand-in keep their spelling. The stand-in lived here
  until 2026-11-18, when it was promoted to ``kinds/meet_moving_point.py`` with its motion unchanged.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import scenario
from fixtures import robots
from robot_trajectories import JointSpace, LinearMotion
from robot_trajectories import meet_moving_point  # noqa: F401  (re-exported: see the module docstring)

FIVEBAR_START = np.radians([128.0, -126.0])  # tests/test_cartesian.py's: the serial elbow at -60°, fold margin 1.0
FIVEBAR_UP = (0.0, -1.0, 0.0)  # WHY: the plane frame's y-hat hangs downward (fb_bench/mapping.py)


def ur5e() -> JointSpace:
    return scenario.ur5e_space().with_kinematics(robots.UR5eKinematics())


def fivebar() -> JointSpace:
    return JointSpace.from_limits(robots.FIVEBAR_LIMITS, home=np.radians([100.0, -76.0])).with_kinematics(robots.FiveBarKinematics())


def linear_target(pose, direction, speed, **extra) -> dict:
    """A constant-velocity target as a lane-F kind's ``target=`` (the recipe keeps it as this mapping)."""
    return dataclasses.asdict(LinearMotion(pose, tuple(direction), float(speed), **extra))
