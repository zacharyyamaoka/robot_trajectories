"""Moving targets: the contract synchronised kinds accept (catalog lane F), and the ruler that holds them to it.

Zach (2026-11-18): "Other things we can put into this library are like sync... You can account for like trajectory points that
are moving". No task words here: the library never knows what moves (a conveyor, a turntable, another robot, a tracked
object). A moving target is a pose that is a function of time, with what its producer knows about it. Kinds see only this;
producers (an encoder-driven belt, a vision tracker's prediction) build it.

**A MovingTarget** (``as_moving_target`` turns a callable, a robot_msgs PoseTrajectory or a fixed pose into one):

| field / method | meaning | bam_traj_gen today |
|---|---|---|
| ``target(t) -> 4x4`` | where it is at time t (s, the trajectory's own clock, t = 0 its first sample), base frame | the authored pose + ``sync_dir`` x ``sync_speed`` x t |
| ``velocity(t) -> (6,)`` | its twist (linear m/s, angular rad/s) in the base frame: what a synchronised leg matches | ``sync_speed`` along ``sync_dir`` |
| ``stamp``, ``latency_s`` | when its state was registered on that clock, and how old that measurement already was | ``execution_latency`` |
| ``window = (enter, leave)`` | the times between which it may be met; a plan outside it is refused | none (reach-limited only) |
| ``assumes`` | the motion model the prediction relies on, in words ("constant velocity 0.25 m/s along +y") | implicit |

**As text**: a recipe is JSON, so a kind's ``target=`` is recorded by its data form (``target_record``, tagged
``moving_target``): a ``LinearMotion``'s fields, or a ``PoseTrajectory`` in robot_msgs' own ``to_dict`` encoding. Both rebuild
the identical object (``target_from_record``), so ``rebuild()`` makes the identical samples. A bare callable has no data form
and is refused by the recipe, naming ``LinearMotion``.

**Which waypoints chase it**: a ``TargetWaypoint`` is a pose in the target's frame plus ``sync``: ``"chase"`` (composed with
``target(t)`` at its own arrival time: it moves with the target), ``"inherit"`` (keeps the offset reached at the sync point
but stops chasing: a leg after contact), ``"fixed"`` (in the base frame). bam_traj_gen's ``ArmParams`` (waypoint_action.py:223)
maps on: ``sync_speed > 0`` is chase, ``sync_at_target_time`` is inherit, otherwise fixed.

**Named refusals**: ``CannotCatch(cause, detail)``, code ``"cannot_catch"`` (bam_traj_gen traj_generator.py:41, Zach
2026-11-11: "Don't hide from the fact you cannot catch it ... make it a known error message"), with causes
``target_outruns_arm``, ``residual_grows``, ``intercept_out_of_reach``, ``time_law_error`` (bam's ``ruckig_error``),
``target_leaves_window``; and ``sync_not_converged``.

**What "synchronised" means**, between ``t0`` and ``t1`` (the sync window): the tool is within ``tolerance_m`` of the target at
every sample with t0 <= t <= t1 (forward kinematics of the output; orientation too when the kinematics places poses), and at
t0 the tool's velocity matches the target's within ``velocity_tolerance_m_s``: it arrives moving with the target, not
catching it up. ``synchronised(space, trajectory, target, t0=..., t1=...)`` is the test every lane-F kind is held to.

**A precomputed synchronised trajectory is a prediction**, made under ``assumes``. Every robot vendor tracks the target live
in the controller (UR adds the conveyor offset each cycle; ABB keeps the tool's speed relative to the moving work object).
So a lane-F kind writes ``assumes`` into its segment's entry, and execution must either replan on new measurements (a
JointTrajectoryController replaces the running trajectory at a bridge point) or apply a controller-side offset; a monitor
can check the assumption against the encoder.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from robot_msgs.geometry.rotations import pose_to_matrix
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory

from .cartesian import rotation_angle
from .space import JointSpace

CANNOT_CATCH = "cannot_catch"
SYNC_NOT_CONVERGED = "sync_not_converged"
CAUSES = ("target_outruns_arm", "residual_grows", "intercept_out_of_reach", "time_law_error", "target_leaves_window")
Sync = Literal["chase", "inherit", "fixed"]
TARGET_TAG = "moving_target"  # the key a moving target's data form is tagged with in a recipe ({"moving_target": "LinearMotion", ...})
DATA_FORMS = ("LinearMotion", "PoseTrajectory")  # the moving targets a recipe can write as text, and rebuild


class MovingTarget(Protocol):
    """A pose that moves (see the module docstring's table)."""

    stamp: float
    latency_s: float
    window: tuple[float, float]
    assumes: str

    def target(self, t: float) -> np.ndarray: ...

    def velocity(self, t: float) -> np.ndarray: ...


class CannotCatch(ValueError):
    """A moving target the plan cannot meet: ``code`` "cannot_catch", ``cause`` one of ``CAUSES``, and the evidence."""

    code = CANNOT_CATCH

    def __init__(self, cause: str, detail: str = ""):
        if cause not in CAUSES:
            raise ValueError(f"cause {cause!r} is not one of {CAUSES}")
        self.cause, self.detail = cause, detail
        super().__init__(f"cannot catch the moving target ({cause.replace('_', ' ')}){': ' + detail if detail else ''}")


def _as_pose(value: Any) -> np.ndarray:
    array = np.asarray(value, float)
    if array.shape == (3,):
        pose = np.eye(4)
        pose[:3, 3] = array
        return pose
    if array.shape != (4, 4):
        raise ValueError(f"a target pose is a 4x4 or an xyz point, got shape {array.shape}")
    return array


@dataclass(frozen=True)
class LinearMotion:
    """A target moving at constant velocity along a line: the common case (a straight conveyor at constant speed).

    ``pose`` is where it was measured; that measurement is ``latency_s`` old at ``stamp`` (on the trajectory's clock), so at
    time t it has moved ``speed * (t - stamp + latency_s)`` along ``direction``."""

    pose: np.ndarray
    direction: tuple[float, float, float]
    speed: float  # m/s
    stamp: float = 0.0
    latency_s: float = 0.0
    window: tuple[float, float] = (-math.inf, math.inf)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", _as_pose(self.pose))
        direction = np.asarray(self.direction, float)
        if self.speed < 0 or not math.isfinite(self.speed):
            raise ValueError(f"speed must be a finite non-negative number, got {self.speed}")
        if self.speed > 0:
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                raise ValueError("a moving target needs a nonzero direction")
            direction = direction / norm
        object.__setattr__(self, "direction", tuple(float(v) for v in direction))

    @classmethod
    def from_sync(cls, pose: Any, sync_dir: Any, sync_speed: float, *, stamp: float = 0.0, execution_latency: float = 0.0,
                  window: tuple[float, float] = (-math.inf, math.inf)) -> LinearMotion:
        """bam_traj_gen's belt chase (``ArmParams.sync_dir`` / ``sync_speed``, ``execution_latency``) as a MovingTarget."""
        return cls(pose, tuple(sync_dir), float(sync_speed), stamp=stamp, latency_s=execution_latency, window=window)

    @property
    def assumes(self) -> str:
        axis = ", ".join(f"{v:+.3f}" for v in self.direction)
        return f"constant velocity {self.speed:.4g} m/s along ({axis}), no rotation"

    def target(self, t: float) -> np.ndarray:
        pose = self.pose.copy()
        pose[:3, 3] += np.asarray(self.direction) * self.speed * (t - self.stamp + self.latency_s)
        return pose

    def velocity(self, t: float) -> np.ndarray:
        return np.concatenate([np.asarray(self.direction) * self.speed, np.zeros(3)])

    def to_record(self) -> dict[str, Any]:
        """Its data form, as a recipe keeps it: every field (defaults written out) under the tag ``moving_target``."""
        return {TARGET_TAG: "LinearMotion", "pose": [[float(v) for v in row] for row in self.pose],
                "direction": [float(v) for v in self.direction], "speed": float(self.speed), "stamp": float(self.stamp),
                "latency_s": float(self.latency_s), "window": [float(v) for v in self.window]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> LinearMotion:
        """The LinearMotion ``to_record`` wrote, field for field: the same object, so a rebuilt recipe makes the same samples."""
        fields = {key: value for key, value in record.items() if key != TARGET_TAG}
        motion = cls(fields["pose"], tuple(fields["direction"]), float(fields["speed"]), stamp=float(fields.get("stamp", 0.0)),
                     latency_s=float(fields.get("latency_s", 0.0)), window=tuple(float(v) for v in fields.get("window", (-math.inf, math.inf))))
        stored = tuple(float(v) for v in fields["direction"])
        if motion.speed > 0 and abs(math.hypot(*stored) - 1.0) > 1e-9:
            raise ValueError(f"a recorded LinearMotion's direction is a unit vector; {stored} is not")
        # WHY the stored direction put back: it was normalised when the target was first made, and normalising it again moves
        # its last bit in about a third of cases (measured 2026-11-18), which would make rebuild() a different trajectory
        object.__setattr__(motion, "direction", stored)
        return motion


@dataclass(frozen=True)
class _Adapted:
    """A callable, PoseTrajectory or fixed pose, wearing the MovingTarget protocol (velocity by central differences)."""

    pose_at: Callable[[float], np.ndarray]
    stamp: float = 0.0
    latency_s: float = 0.0
    window: tuple[float, float] = (-math.inf, math.inf)
    assumes: str = "the target function as given"

    def target(self, t: float) -> np.ndarray:
        if not self.window[0] - 1e-12 <= t <= self.window[1] + 1e-12:
            raise CannotCatch("target_leaves_window", f"asked at t = {t:.3f} s, the window is {self.window[0]:.3f}..{self.window[1]:.3f} s")
        return _as_pose(self.pose_at(t))

    def velocity(self, t: float) -> np.ndarray:
        h = 1e-4
        low, high = max(t - h, self.window[0]), min(t + h, self.window[1])
        a, b = self.target(low), self.target(high)
        linear = (b[:3, 3] - a[:3, 3]) / (high - low)
        delta = a[:3, :3].T @ b[:3, :3]
        angle = rotation_angle(a[:3, :3], b[:3, :3])
        axis = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
        norm = float(np.linalg.norm(axis))
        angular = np.zeros(3) if norm < 1e-12 else a[:3, :3] @ (axis / norm) * angle / (high - low)
        return np.concatenate([linear, angular])


def as_moving_target(target: Any) -> MovingTarget:
    """Any accepted form as a MovingTarget: one already, a robot_msgs PoseTrajectory (its window is its span), a callable
    ``target(t) -> pose``, or a fixed pose or point (a target that does not move)."""
    if callable(getattr(target, "target", None)) and callable(getattr(target, "velocity", None)):
        return target
    if isinstance(target, PoseTrajectory):
        start, end = float(target.t[0]), float(target.t[-1])

        def from_trajectory(t: float) -> np.ndarray:
            point = target.at(min(max(t, start), end))
            return pose_to_matrix(point.position, point.orientation)

        return _Adapted(from_trajectory, window=(start, end), assumes="the PoseTrajectory as given (no extrapolation)")
    if callable(target):
        return _Adapted(target)
    fixed = _as_pose(target)
    return _Adapted(lambda t: fixed, assumes="a fixed pose")


# ====================================================================== a moving target as text (what a recipe keeps)


def has_data_form(value: Any) -> bool:
    """Whether a recipe can write ``value`` out as a moving target and rebuild it: a LinearMotion or a robot_msgs PoseTrajectory."""
    return isinstance(value, (LinearMotion, PoseTrajectory))


def target_record(target: Any) -> dict[str, Any]:
    """A moving target's data form, tagged ``moving_target``: a LinearMotion's fields, or a PoseTrajectory in robot_msgs' own
    dict encoding (``to_dict``: every sample, dtype and shape kept). A function of time has none, and is refused.

    WHY only these two: a recipe must rebuild the identical target from text, and a callable is code, not data."""
    if isinstance(target, LinearMotion):
        return target.to_record()
    if isinstance(target, PoseTrajectory):
        return {TARGET_TAG: "PoseTrajectory", "trajectory": target.to_dict()}
    raise TypeError(f"{target!r} has no data form: a recipe keeps a moving target as a LinearMotion (constant velocity; "
                    "LinearMotion.from_sync(pose, sync_dir, sync_speed)) or a robot_msgs PoseTrajectory (sampled poses), "
                    "never a function of time")


def is_target_record(value: Any) -> bool:
    return isinstance(value, Mapping) and TARGET_TAG in value


def target_from_record(record: Mapping[str, Any]) -> LinearMotion | PoseTrajectory:
    """The moving target ``target_record`` wrote: identical, so the kind that reads it makes the identical samples again."""
    form = record[TARGET_TAG]
    if form == "LinearMotion":
        return LinearMotion.from_record(record)
    if form == "PoseTrajectory":
        return PoseTrajectory.from_dict(record["trajectory"])
    raise ValueError(f"a moving target recorded as {form!r}: this version reads {DATA_FORMS}")


@dataclass(frozen=True)
class TargetWaypoint:
    """A waypoint written in the moving target's frame, and whether it chases the target (see the module docstring)."""

    pose_in_target: Any  # 4x4, or an xyz offset
    sync: Sync = "chase"
    dwell_s: float = 0.0

    def __post_init__(self) -> None:
        if self.sync not in ("chase", "inherit", "fixed"):
            raise ValueError(f"sync is 'chase', 'inherit' or 'fixed', got {self.sync!r}")
        object.__setattr__(self, "pose_in_target", _as_pose(self.pose_in_target))

    @classmethod
    def from_arm_params(cls, pose_in_target: Any, *, sync_speed: float, sync_at_target_time: bool, dwell_s: float = 0.0) -> TargetWaypoint:
        """bam_traj_gen's ``ArmParams`` sync fields as a TargetWaypoint."""
        return cls(pose_in_target, "chase" if sync_speed > 0 else "inherit" if sync_at_target_time else "fixed", dwell_s)


@dataclass(frozen=True)
class SyncReport:
    """How well a trajectory tracks a moving target over its sync window."""

    synchronised: bool
    worst_position_error_m: float
    worst_at_s: float
    worst_orientation_error_rad: float
    velocity_error_m_s: float  # |tool velocity - target velocity| at t0
    samples: int
    why: str = ""

    def __bool__(self) -> bool:
        return self.synchronised


def synchronised(space: JointSpace, trajectory: JointTrajectory, target: Any, *, t0: float, t1: float, tolerance_m: float = 0.001,
                 tolerance_rad: float = math.radians(1.0), velocity_tolerance_m_s: float = 0.01) -> SyncReport:
    """Whether ``trajectory`` tracks ``target`` from ``t0`` to ``t1`` (the module docstring's definition)."""
    moving = as_moving_target(target)
    t = np.asarray(trajectory.t, float)
    window = np.flatnonzero((t >= t0 - 1e-9) & (t <= t1 + 1e-9))
    if len(window) < 2:
        return SyncReport(False, math.inf, t0, math.inf, math.inf, len(window), "the sync window holds fewer than two samples")
    position_errors, orientation_errors = [], []
    for row in window:
        reached, wanted = space.forward(trajectory.positions[row]), moving.target(float(t[row]))
        position_errors.append(float(np.linalg.norm(reached[:3, 3] - wanted[:3, 3])))
        orientation_errors.append(0.0 if space.position_only else rotation_angle(reached[:3, :3], wanted[:3, :3]))
    first, second = window[0], window[1]
    dt = float(t[second] - t[first])
    tool_velocity = (space.forward(trajectory.positions[second])[:3, 3] - space.forward(trajectory.positions[first])[:3, 3]) / dt
    velocity_error = float(np.linalg.norm(tool_velocity - np.asarray(moving.velocity(float(t[first])))[:3]))
    worst = int(np.argmax(position_errors))
    why = []
    if position_errors[worst] > tolerance_m:
        why.append(f"the tool is {1000 * position_errors[worst]:.3f} mm from the target at t = {t[window[worst]]:.3f} s "
                   f"(tolerance {1000 * tolerance_m:.3f} mm)")
    if max(orientation_errors) > tolerance_rad:
        why.append(f"orientation off by {math.degrees(max(orientation_errors)):.3f}°")
    if velocity_error > velocity_tolerance_m_s:
        why.append(f"at the sync point the tool's velocity differs from the target's by {velocity_error:.4f} m/s "
                   f"(tolerance {velocity_tolerance_m_s:g})")
    return SyncReport(not why, position_errors[worst], float(t[window[worst]]), max(orientation_errors), velocity_error,
                      len(window), "; ".join(why))


__all__ = ["CANNOT_CATCH", "CAUSES", "CannotCatch", "DATA_FORMS", "LinearMotion", "MovingTarget", "SYNC_NOT_CONVERGED", "SyncReport",
           "TARGET_TAG", "TargetWaypoint", "as_moving_target", "has_data_form", "is_target_record", "synchronised", "target_from_record",
           "target_record"]
