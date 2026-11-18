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


# ====================================================================== kinematics, dynamics and a scene to inject (numpy only)
# Added for robot_trajectories 0.1.0 (PEP 0002): the shapes a JointSpace takes by duck typing, never by import. Stand-ins,
# labelled: the five-bar's closed form is the bench's geometry above; the UR5e's is its published DH table; the pendulum's
# dynamics and the UR5e scene are made-up test values, not measurements.


def fivebar_tool_point(theta_j2: float, theta_c1: float) -> tuple[float, float] | None:
    """The five-bar's tool point (x, y) in its plane frame (m) for a motor pair, or None where the linkage does not close
    (fb_bench/mapping.py ee_position on fivebar_passive's j3)."""
    passive = fivebar_passive(theta_j2, theta_c1)
    if passive is None:
        return None
    g = FIVEBAR
    j3 = passive[0]
    return (g["L_shoulder"] + g["L_l1"] * math.cos(theta_j2) + g["L_l2"] * math.cos(theta_j2 + j3),
            g["L_l1"] * math.sin(theta_j2) + g["L_l2"] * math.sin(theta_j2 + j3))


@dataclass(frozen=True)
class FiveBarKinematics:
    """The bench five-bar's tool point, closed form both ways (bam_reach's Kinematics shape; position only: a planar point).

    ``inverse``: the serial chain's elbow from the point (j3 in either branch), then the crank angle where the coupler closes
    (two circle intersections), kept only where ``forward`` of the answer lands on the point with the fold margin at least
    ``min_fold_margin`` (the scenario's reachability test uses 0.2), nearest the seed."""

    name: str = "bam_fb bench five-bar, tool point (closed form)"
    joint_names: tuple[str, ...] = ("j2", "c1")
    cache_key: str = "fivebar-closed-form-1"
    orientation_constraint: str = "none"
    min_fold_margin: float = 0.2

    def forward(self, joint_positions):
        j2, c1 = (float(value) for value in joint_positions)
        passive = fivebar_passive(j2, c1)
        if passive is None or passive[2] < self.min_fold_margin:
            return None
        x, y = fivebar_tool_point(j2, c1)
        heading = j2 + passive[0]
        pose = np.eye(4)
        pose[:2, :2] = [[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]]
        pose[:2, 3] = (x, y)
        return pose

    def inverse(self, target_pose, seed=None):
        target = np.asarray(target_pose, float)
        x, y = (target[:3, 3] if target.shape == (4, 4) else target)[:2]
        g = FIVEBAR
        dx, dy = x - g["L_shoulder"], y
        cosine = (dx * dx + dy * dy - g["L_l1"] ** 2 - g["L_l2"] ** 2) / (2 * g["L_l1"] * g["L_l2"])
        if abs(cosine) > 1:
            return None
        found = []
        for j3 in (-math.acos(cosine), math.acos(cosine)):
            j2 = _wrap(math.atan2(dy, dx) - math.atan2(g["L_l2"] * math.sin(j3), g["L_l1"] + g["L_l2"] * math.cos(j3)))
            lever = j2 + j3 + math.pi - g["theta_elbow_offset"]
            px = g["L_shoulder"] + g["L_l1"] * math.cos(j2) + g["L_elbow"] * math.cos(lever)
            py = g["L_l1"] * math.sin(j2) + g["L_elbow"] * math.sin(lever)
            rho = math.hypot(px, py)
            reach = (g["L_c1"] ** 2 + rho * rho - g["L_c2"] ** 2) / (2 * g["L_c1"] * rho)
            if abs(reach) > 1:
                continue
            for c1 in (math.atan2(py, px) + math.acos(reach), math.atan2(py, px) - math.acos(reach)):
                c1 = _wrap(c1)
                if self.forward((j2, c1)) is None:
                    continue
                point = fivebar_tool_point(j2, c1)
                if math.hypot(point[0] - x, point[1] - y) < 1e-9:
                    found.append(np.array([j2, c1]))
        if not found:
            return None
        if seed is None:
            return found[0]
        seed = np.asarray(seed, float)
        return min(found, key=lambda q: float(np.max(np.abs([_wrap(a - b) for a, b in zip(q, seed)]))))


# UR5e, standard DH (Universal Robots' published parameters): link offsets d, lengths a, twists alpha.
UR5E_DH_D = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
UR5E_DH_A = (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0)
UR5E_DH_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)


def ur5e_frames(q) -> list[np.ndarray]:
    """The base frame and each of the six DH frames, as 4x4 transforms in the base frame (the last is the tool flange)."""
    frames = [np.eye(4)]
    for theta, d, a, alpha in zip(np.asarray(q, float), UR5E_DH_D, UR5E_DH_A, UR5E_DH_ALPHA, strict=True):
        ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
        frames.append(frames[-1] @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st],
                                             [0.0, sa, ca, d], [0.0, 0.0, 0.0, 1.0]]))
    return frames


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    skew = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0)))
    return 0.5 * skew if angle < 1e-9 else angle / (2.0 * math.sin(angle)) * skew


@dataclass(frozen=True)
class UR5eKinematics:
    """The UR5e's flange pose: DH forward kinematics, damped-least-squares inverse kinematics seeded at ``seed``."""

    name: str = "UR5e (DH forward, damped least squares inverse)"
    joint_names: tuple[str, ...] = UR5E_JOINTS
    cache_key: str = "ur5e-dh-dls-1"
    orientation_constraint: str = "full"
    damping: float = 0.01
    max_iterations: int = 200

    def forward(self, joint_positions):
        return ur5e_frames(joint_positions)[-1]

    def jacobian(self, joint_positions) -> np.ndarray:
        frames = ur5e_frames(joint_positions)
        tool = frames[-1][:3, 3]
        columns = []
        for frame in frames[:-1]:
            axis = frame[:3, 2]
            columns.append(np.concatenate([np.cross(axis, tool - frame[:3, 3]), axis]))
        return np.array(columns).T

    def inverse(self, target_pose, seed=None):
        target = np.asarray(target_pose, float)
        q = np.zeros(6) if seed is None else np.array(seed, float)
        for _ in range(self.max_iterations):
            pose = self.forward(q)
            error = np.concatenate([target[:3, 3] - pose[:3, 3], _rotation_vector(target[:3, :3] @ pose[:3, :3].T)])
            if np.linalg.norm(error[:3]) < 1e-10 and np.linalg.norm(error[3:]) < 1e-9:
                return q
            jacobian = self.jacobian(q)
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + self.damping**2 * np.eye(6), error)
            largest = float(np.max(np.abs(step)))
            q = q + (step * 0.2 / largest if largest > 0.2 else step)
        return None


@dataclass(frozen=True)
class UR5eScene:
    """A test scene for the UR5e: a floor plane and one axis-aligned box, checked at the DH frame origins from the elbow out
    and the midpoints between them (``in_collision`` and ``distance``, the CollisionChecker shape). Made-up geometry."""

    box_min: tuple[float, float, float] = (0.6, 0.6, 0.0)
    box_max: tuple[float, float, float] = (0.8, 0.8, 0.2)
    floor_z: float = 0.0
    name: str = "floor plane + one box (test stand-in)"
    cache_key: str = "ur5e-scene-1"

    def points(self, q) -> np.ndarray:
        origins = np.array([frame[:3, 3] for frame in ur5e_frames(q)[1:]])
        return np.vstack([origins[1:], 0.5 * (origins[:-1] + origins[1:])])

    def distance(self, q) -> float:
        points = self.points(q)
        low, high = np.asarray(self.box_min), np.asarray(self.box_max)
        outside = np.linalg.norm(np.maximum(np.maximum(low - points, points - high), 0.0), axis=1)
        inside = np.min(np.minimum(points - low, high - points), axis=1)
        to_box = np.where(outside > 0, outside, -inside)
        return float(min(np.min(points[:, 2] - self.floor_z), np.min(to_box)))

    def in_collision(self, q) -> bool:
        return self.distance(q) < 0.0


@dataclass(frozen=True)
class PendulumDynamics:
    """The pendulum as J qdd + b qd + m g l sin(q) = tau, q from the gravity bottom (made-up values: a stand-in for lane B's
    effort-commanded kinds, not CAN 11's identified model)."""

    inertia: float = 0.02  # kg m^2
    gravity_torque: float = 0.3  # m g l, N m
    damping: float = 0.05  # N m s / rad
    name: str = "pendulum stand-in (J qdd + b qd + m g l sin q)"
    cache_key: str = "pendulum-stand-in-1"

    def torque(self, q, qd, qdd):
        q, qd, qdd = (np.asarray(value, float) for value in (q, qd, qdd))
        return self.inertia * qdd + self.damping * qd + self.gravity_torque * np.sin(q)

    def forward(self, q, qd, tau):
        q, qd, tau = (np.asarray(value, float) for value in (q, qd, tau))
        return (tau - self.damping * qd - self.gravity_torque * np.sin(q)) / self.inertia
