# The round's fixture, unchanged: bam_ws docs/trajectory_generator/shared/robots.py at b5e1f6965. The CAN 11 limits, the
# five-bar's geometry and the UR5e box are instance values: they stay here (and in examples/), never in the package.
"""The three robots every proposal and landing page uses, each handed over in the form Zach named as an entry point.

- **1-DOF pendulum** (the One Dof Test Rack's CAN 11 station) as **bare joint limits**: the values in its schedule document
  (`/archive/datasets/bam_rig/schedules/one-dof-test-rack/can11/schedule.json`): +/-0.33 rad about the gravity bottom,
  2.0 rad/s, 7.39 rad/s^2. Positions are relative to the origin, as sim2real's schedule writes them.
- **2-DOF five-bar** (bam_fb's two drives, J2 and C1) as a **reach map**: a `ReachMapLike` whose valid joint solutions
  are the motor pairs where the linkage closes with a fold margin (closed-form, the bench's `fb_bench/mapping.py`
  geometry copied here so this file stays numpy-only). Shaped like `bam_reach.ReachMap`: `joint_names`,
  `joint_solutions` (P, O, J), `valid_mask` (P, O). A ReachMap carries no velocity or acceleration, so it travels
  with the description's limits.
- **6-DOF UR5e** as a **robot description**: a stand-in with `bam_robot_descriptions.RobotDescription.joint_limits()`'s
  signature and its real gap: the URDF has position and velocity limits but **no acceleration** (urdf.py:1430).

Stand-ins, labelled: the five-bar reach map is computed here (not loaded from a `.reach.npz`), and the UR5e
description is a stand-in object, because the generator core may not import bam_reach or bam_robot_descriptions
(the leaf rule) and this fixture must run in any proposal's interpreter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from robot_msgs.trajectory import JointLimits

# ------------------------------------------------------------------ 1-DOF: bare joint limits

PENDULUM_LIMITS = JointLimits(
    joint_names=("can11",),
    min_position=np.array([-0.33]),
    max_position=np.array([0.33]),
    max_velocity=np.array([2.0]),
    max_acceleration=np.array([7.3896]),
    position_kind=("software",),
)
PENDULUM_ORIGIN_RAD = -0.373685  # the gravity bottom in the board frame; positions above are relative to it

# ------------------------------------------------------------------ 2-DOF: a reach map

FIVEBAR = dict(L_l1=0.240, L_l2=0.299658, L_c1=0.120, L_c2=0.260, L_shoulder=0.087, L_elbow=0.120, theta_elbow_offset=math.pi / 4)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def fivebar_passive(theta_j2: float, theta_c1: float, *, j3_hint: float = -1.6) -> tuple[float, float, float] | None:
    """(j3, c2, fold_margin) for a motor pair, the branch nearest ``j3_hint`` (copied from fb_bench/mapping.solve_passive)."""
    g = FIVEBAR
    phi = math.pi - g["theta_elbow_offset"]
    e0 = (g["L_shoulder"] + g["L_l1"] * math.cos(theta_j2), g["L_l1"] * math.sin(theta_j2))
    crank = (g["L_c1"] * math.cos(theta_c1), g["L_c1"] * math.sin(theta_c1))
    dx, dy = e0[0] - crank[0], e0[1] - crank[1]
    d = math.hypot(dx, dy)
    if d < 1e-12:
        return None
    cos_arg = (g["L_c2"] ** 2 - g["L_elbow"] ** 2 - d * d) / (2 * g["L_elbow"] * d)
    if abs(cos_arg) > 1:
        return None
    base, delta = math.atan2(dy, dx), math.acos(max(-1.0, min(1.0, cos_arg)))
    best, best_distance = None, math.inf
    for sign in (1.0, -1.0):
        psi = base + sign * delta
        j3 = _wrap(psi - phi - theta_j2)
        pin = (e0[0] + g["L_elbow"] * math.cos(psi), e0[1] + g["L_elbow"] * math.sin(psi))
        rod = math.atan2(pin[1] - crank[1], pin[0] - crank[0])
        candidate = (j3, _wrap(rod - theta_c1), abs(math.sin(psi - rod)))
        if abs(_wrap(j3 - j3_hint)) < best_distance:
            best, best_distance = candidate, abs(_wrap(j3 - j3_hint))
    return best


@dataclass(frozen=True)
class ReachMapLike:
    """The fields of bam_reach.ReachMap a trajectory generator reads (reach_map.py): names, solutions, validity."""

    joint_names: tuple[str, ...]
    joint_solutions: np.ndarray  # (P, O, J)
    valid: np.ndarray  # (P, O)
    kinematics_name: str = ""

    @property
    def valid_mask(self) -> np.ndarray:
        return self.valid

    def reachable_configurations(self) -> np.ndarray:
        """(M, J) every valid joint solution, flattened over poses and orientations."""
        return self.joint_solutions[self.valid]


def fivebar_reach_map(step_deg: float = 4.0, min_fold_margin: float = 0.35) -> ReachMapLike:
    """Motor pairs (J2, C1) on a grid where the linkage closes away from a fold: the 2-DOF's reachable set."""
    j2_values = np.radians(np.arange(40.0, 141.0, step_deg))
    c1_values = np.radians(np.arange(-160.0, 21.0, step_deg))
    solutions, valid = [], []
    for j2 in j2_values:
        for c1 in c1_values:
            passive = fivebar_passive(float(j2), float(c1))
            solutions.append([[j2, c1]])
            valid.append([passive is not None and passive[2] >= min_fold_margin])
    return ReachMapLike(("j2", "c1"), np.array(solutions), np.array(valid), kinematics_name="bam_fb bench five-bar (closed form)")


FIVEBAR_LIMITS = JointLimits(  # the description's drive limits; positions are the reach map's bounding box
    joint_names=("j2", "c1"),
    min_position=np.radians([40.0, -160.0]),
    max_position=np.radians([140.0, 20.0]),
    max_velocity=np.array([3.0, 3.0]),
    max_acceleration=np.array([12.0, 12.0]),
)

# ------------------------------------------------------------------ 6-DOF: a robot description

UR5E_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


@dataclass(frozen=True)
class RobotDescriptionLike:
    """Stand-in with bam_robot_descriptions.RobotDescription.joint_limits()'s signature and its URDF-only content."""

    name: str
    _limits: JointLimits

    def joint_limits(self, names: tuple[str, ...] | list[str] | None = None) -> JointLimits:
        return self._limits if names is None else self._limits.select(tuple(names))


UR5E_DESCRIPTION = RobotDescriptionLike(
    name="ur5e",
    _limits=JointLimits(  # ur_description's URDF values: +/-360 deg, 180 deg/s, and no acceleration (as every URDF)
        joint_names=UR5E_JOINTS,
        min_position=np.full(6, -2 * math.pi),
        max_position=np.full(6, 2 * math.pi),
        max_velocity=np.full(6, math.pi),
        max_acceleration=None,
    ),
)
# A demo box above the table with the elbow up, so previews never fold the arm into its base.
UR5E_DEMO_BOX = JointLimits(
    joint_names=UR5E_JOINTS,
    min_position=np.radians([-90.0, -130.0, 40.0, -150.0, -120.0, -90.0]),
    max_position=np.radians([90.0, -50.0, 130.0, -40.0, -60.0, 90.0]),
    max_velocity=np.full(6, math.pi),
    max_acceleration=np.full(6, 6.0),
    position_kind=("software",) * 6,
)
UR5E_ASSUMED_ACCELERATION = 6.0  # rad/s^2: assumed for previews (the URDF has none)
