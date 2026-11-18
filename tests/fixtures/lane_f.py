"""Lane F's fixtures: the two robots with kinematics, targets moving at constant velocity, and one TEST STAND-IN kind.

- ``ur5e()`` / ``fivebar()``: the spaces tests/test_cartesian.py uses (the UR5e's demo box with its DH kinematics; the five-bar's
  limits with its closed-form tool point), and ``FIVEBAR_START``, inside its reach away from the elbow singularity.
- ``linear_target(...)``: a ``LinearMotion`` as the mapping a lane-F kind takes (and its recipe keeps), ``dataclasses.asdict``.
- ``meet_moving_point``: a STAND-IN kind (no catalog row; registered only where this module is imported): from rest, meet a
  target and stay on it, ENDING MOVING with it. No catalog row ends moving at a target, so this is what the tests chain
  ``track_and_release`` after. Built from the same helpers as ``moving_target_track`` (its first two thirds).
"""

from __future__ import annotations

import dataclasses

import numpy as np

import scenario
from fixtures import robots
from robot_trajectories import JointSpace, LinearMotion, Motion, TimeLaw, kind
from robot_trajectories.kinds import _lane_f as f
from robot_trajectories.registry import LIMITS

FIVEBAR_START = np.radians([128.0, -126.0])  # tests/test_cartesian.py's: the serial elbow at -60°, fold margin 1.0
FIVEBAR_UP = (0.0, -1.0, 0.0)  # WHY: the plane frame's y-hat hangs downward (fb_bench/mapping.py)


def ur5e() -> JointSpace:
    return scenario.ur5e_space().with_kinematics(robots.UR5eKinematics())


def fivebar() -> JointSpace:
    return JointSpace.from_limits(robots.FIVEBAR_LIMITS, home=np.radians([100.0, -76.0])).with_kinematics(robots.FiveBarKinematics())


def linear_target(pose, direction, speed, **extra) -> dict:
    """A constant-velocity target as a lane-F kind's ``target=`` (the recipe keeps it as this mapping)."""
    return dataclasses.asdict(LinearMotion(pose, tuple(direction), float(speed), **extra))


@kind(use_cases=("tracking_eval",), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"), family="task_cycle",
      starts_as="path", dimensions=("2D", "3D", "6D"), ends="moving")
def meet_moving_point(space, target=None, *, sync_start_s=1.0, sync_end_s=1.3, time_law=TimeLaw.minimum_jerk(peak_speed_fraction=0.5),
                      label="meet a moving point", start=None):
    """TEST STAND-IN (tests/fixtures/lane_f.py, no catalog row): from rest, meet ``target`` at ``sync_start_s`` with its
    velocity matched and stay on it to ``sync_end_s``, ending there moving with it (the tool holds its start orientation).
    ``target=None``: moving_target_track's default."""
    rate = space.rate_hz
    first, last = round(sync_start_s * rate), round(sync_end_s * rate)
    moving = f.moving_target(f.default_target(space, start, 0.5 * (first + last) / rate) if target is None else target)
    pieces, notes, _ = f.meet(space, start, moving, first, last, rotation=space.forward(start)[:3, :3], offset=np.zeros(3),
                              time_law=time_law, label=label)
    return Motion(pieces, notes, {"assumes": moving.assumes, "sync": [first / rate, last / rate]})
