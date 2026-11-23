"""Lane F's shared helpers: what the moving-target kinds (``moving_target_track``, ``track_and_release``,
``moving_pick_and_place``) are built on. Imported by those kinds; registers nothing (KINDS_TODO "Rules of the road").

- ``moving_target(value)``: a kind's ``target=`` as a MovingTarget: a ``LinearMotion`` or ``PoseTrajectory`` object (the
  recipe keeps its data form), a mapping of ``LinearMotion``'s fields (``dataclasses.asdict(LinearMotion.from_sync(pose,
  sync_dir, sync_speed))`` is one), or a fixed xyz / 4x4 (a target that does not move). ``on_clock`` reads a target with a
  kind's t = 0 somewhere else on the target's clock.
- ``ToolPath`` and its makers: ``chase`` (the tool on the target plus an offset that may itself move: the descent onto a
  moving point is a ramped offset) and ``coast`` (from moving to rest: braking along the way it was going while it travels a
  displacement). A tool path given in time, its velocity and acceleration in closed form.
- ``jacobian(space, q)``: numeric, from ``space.forward`` (central differences). ``follow_tool``: inverse kinematics at every
  sample of a ToolPath, the joint velocity J^+ v and acceleration J^+ (a - Jdot qdot) that give the tool's.
- ``synced``: a chase that must keep pace (``CannotCatch`` when it cannot); ``fitted``: the shortest duration on the rate grid
  at which a made piece stays inside the velocity and acceleration limits; ``approach``: from rest to a moving state at a given
  sample, under a waypoint trajectory generator (the quintic through both states).

Every tool path here holds one orientation: ``target_motion`` refuses a target that rotates (a ``PoseTrajectory`` may)
rather than drop its rotation; ``LinearMotion`` and a fixed pose never rotate.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..cartesian import Unreachable, rotation_angle
from ..moving import CannotCatch, LinearMotion, as_moving_target, is_target_record, target_from_record
from ..segments import TOOL_TARGET, Piece
from ..space import JointSpace, State
from ..waypoint_trajectory_generator import BoundaryState, WaypointTrajectoryGenerator

TARGET_FIELDS = tuple(field.name for field in dataclasses.fields(LinearMotion))
MAX_JOINT_STEP_RAD = 0.1  # WHY 0.1: cartesian.follow's absolute branch-flip threshold between samples
JACOBIAN_STEP_RAD = 1e-6
CURVATURE_STEP_RAD = 1e-4  # WHY: the Jdot qdot probe moves the joints this far along qdot, large against the Jacobian's round-off
DIFFERENCE_STEP_S = 1e-3  # the target's acceleration by central differences of its velocity (exact at constant velocity)
FIT_STRIDE = 10  # WHY: a candidate duration is screened at every 10th sample (50 Hz at 500 Hz), then checked at every one
FIT_MARGIN = 0.98  # WHY: the screen keeps 2 % for the peaks between its samples
FIT_TRIES = 60
DEFAULT_SPEED_M_S = 0.05  # WHY 5 cm/s: half of bam_traj_gen's slowest measured chase (0.1 m/s), so a default is easy to meet
PACE_VELOCITY_TOLERANCE_M_S = 0.01  # WHY: synchronised()'s velocity tolerance: a continued state must already move with the target


# ====================================================================== the target


def moving_target(value: Any) -> Any:
    """``target=`` as a MovingTarget: a ``LinearMotion`` or a robot_msgs ``PoseTrajectory`` (the registry records either by its
    data form and hands the rebuilt object back on ``rebuild``), a mapping of LinearMotion's fields (pose, direction, speed, and
    optionally stamp, latency_s, window: the form these kinds took first, kept), or a fixed xyz / 4x4."""
    if is_target_record(value):
        return as_moving_target(target_from_record(value))
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(TARGET_FIELDS))
        missing = [name for name in ("pose", "direction", "speed") if name not in value]
        if unknown or missing:
            raise ValueError(f"a moving target is a mapping of LinearMotion's fields {TARGET_FIELDS} (pose, direction and speed "
                             f"required){'; unknown: ' + ', '.join(unknown) if unknown else ''}"
                             f"{'; missing: ' + ', '.join(missing) if missing else ''}")
        fields = dict(value)
        fields["direction"] = tuple(float(v) for v in fields["direction"])
        if "window" in fields:
            fields["window"] = tuple(float(v) for v in fields["window"])
        return LinearMotion(**fields)
    return as_moving_target(value)


@dataclass(frozen=True)
class OnClock:
    """A MovingTarget read with this kind's t = 0 at ``offset_s`` on the target's own clock (a kind that continues another)."""

    moving: Any
    offset_s: float

    @property
    def stamp(self) -> float:
        return float(self.moving.stamp) - self.offset_s

    @property
    def latency_s(self) -> float:
        return float(self.moving.latency_s)

    @property
    def window(self) -> tuple[float, float]:
        enter, leave = self.moving.window
        return (enter - self.offset_s, leave - self.offset_s)

    @property
    def assumes(self) -> str:
        return self.moving.assumes

    def target(self, t: float) -> np.ndarray:
        return self.moving.target(t + self.offset_s)

    def velocity(self, t: float) -> np.ndarray:
        return self.moving.velocity(t + self.offset_s)


def default_target(space: JointSpace, start: Any, at_s: float, *, shift: Any = (0.0, 0.0, 0.0), across: Any = None) -> dict:
    """``target=None``: a deterministic constant-velocity target derived from the space, in the recipe's mapping form. At
    ``at_s`` it is at the tool's start point plus ``shift`` (base frame), with the tool's start orientation, moving at
    ``DEFAULT_SPEED_M_S`` along the way the joint that moves the tool most at the start moves it (minus any part along
    ``across``: e.g. a vertical, so the default stays level). WHY that direction: the arm can move the tool along it at the
    start, on any robot (in the plane of a planar one), so the default target crosses the window inside the reach."""
    here = space.forward(start)
    columns = jacobian(space, start)[:3].T
    if across is not None:
        normal = np.asarray(across, float) / np.linalg.norm(across)
        columns = columns - np.outer(columns @ normal, normal)
    column = columns[int(np.argmax(np.linalg.norm(columns, axis=1)))]
    direction = column / np.linalg.norm(column)
    pose = here.copy()
    pose[:3, 3] = here[:3, 3] + np.asarray(shift, float) - direction * DEFAULT_SPEED_M_S * at_s
    return dataclasses.asdict(LinearMotion(pose, tuple(float(v) for v in direction), DEFAULT_SPEED_M_S))


def following(space: JointSpace, state: State) -> dict:
    """``target=None`` for a kind that continues a moving state: the tool's own motion at that state, extrapolated at constant
    velocity (a fixed point where it is, when it is at rest). The only target a start is sure to be moving with."""
    here = space.forward(state.q)
    velocity = (jacobian(space, state.q) @ np.asarray(state.qd, float))[:3]
    speed = float(np.linalg.norm(velocity))
    direction = velocity / speed if speed > 1e-12 else np.zeros(3)
    return dataclasses.asdict(LinearMotion(here, tuple(float(v) for v in direction), speed if speed > 1e-12 else 0.0))


def as_pose(value: Any) -> np.ndarray:
    """An xyz (3,) or a 4x4 as a 4x4 pose."""
    array = np.asarray(value, float)
    if array.shape == (3,):
        pose = np.eye(4)
        pose[:3, 3] = array
        return pose
    if array.shape != (4, 4):
        raise ValueError(f"a pose is a 4x4 or an xyz point, got shape {array.shape}")
    return array


def posed(rotation: Any, point: Any) -> np.ndarray:
    """The 4x4 pose with this rotation at this point."""
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, point
    return pose


def on_clock(target: Any, offset_s: float) -> Any:
    return target if offset_s == 0 else OnClock(target, float(offset_s))


def check_window(target: Any, begin_s: float, end_s: float, what: str) -> None:
    """``CannotCatch("target_leaves_window")`` when ``begin_s..end_s`` is not inside the target's window."""
    enter, leave = target.window
    if begin_s < enter - 1e-9 or end_s > leave + 1e-9:
        raise CannotCatch("target_leaves_window", f"{what} runs t = {begin_s:.3f}..{end_s:.3f} s; the target may be met only "
                                                  f"between t = {enter:.3f} and {leave:.3f} s")


def sample_of(time_s: float, rate_hz: float, name: str) -> tuple[int, str]:
    """A time as a whole sample on the rate grid, and a note when it had to be rounded onto it."""
    exact = time_s * rate_hz
    sample = round(exact)
    note = f"{name} {time_s:g} s rounded to {sample / rate_hz:g} s, the {rate_hz:g} Hz grid" if abs(exact - sample) > 1e-6 else ""
    return sample, note


# ====================================================================== tool paths given in time


@dataclass(frozen=True)
class ToolPath:
    """The tool at ``times`` (N,): positions (N, 3), velocities, accelerations (base frame), one held orientation."""

    times: np.ndarray
    points: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    rotation: np.ndarray

    def pose(self, row: int) -> np.ndarray:
        return posed(self.rotation, self.points[row])


def minimum_jerk(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5, its two derivatives, and its integral from 0 (1/2 at tau = 1)."""
    tau = np.asarray(tau, float)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    integral = 2.5 * tau**4 - 3 * tau**5 + tau**6
    return s, ds, dds, integral


def target_motion(target: Any, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(points, velocities, accelerations, rotation at the first time) of the target at ``times``. Refuses a rotating target."""
    poses = [np.asarray(target.target(float(t)), float) for t in times]
    twists = np.array([np.asarray(target.velocity(float(t)), float) for t in times])
    spin = float(np.max(np.linalg.norm(twists[:, 3:], axis=1)))
    if spin > 1e-9:
        raise ValueError(f"the target rotates ({math.degrees(spin):.3f}°/s): lane F's kinds hold the tool's orientation, so a "
                         "rotating target is not served yet (its rotation would be dropped, not tracked)")
    h = DIFFERENCE_STEP_S
    accelerations = np.array([(np.asarray(target.velocity(float(t) + h), float)[:3] - np.asarray(target.velocity(float(t) - h), float)[:3])
                              / (2 * h) for t in times])
    return np.array([pose[:3, 3] for pose in poses]), twists[:, :3], accelerations, poses[0][:3, :3]


Offset = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]  # times -> (offset, its velocity, its acceleration)


def fixed_offset(vector: Any) -> Offset:
    vector = np.asarray(vector, float)

    def at(times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = len(times)
        return np.tile(vector, (count, 1)), np.zeros((count, 3)), np.zeros((count, 3))

    return at


def ramped_offset(begin: Any, end: Any, begin_s: float, duration_s: float) -> Offset:
    """An offset going from ``begin`` to ``end`` (base frame) over ``duration_s`` from ``begin_s``, minimum-jerk in time."""
    begin, change = np.asarray(begin, float), np.asarray(end, float) - np.asarray(begin, float)

    def at(times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s, ds, dds, _ = minimum_jerk(np.clip((np.asarray(times, float) - begin_s) / duration_s, 0.0, 1.0))
        return begin + change * s[:, None], change * ds[:, None] / duration_s, change * dds[:, None] / duration_s**2

    return at


def chase(target: Any, times: np.ndarray, rotation: np.ndarray, offset: Offset) -> ToolPath:
    """The tool on the moving target plus ``offset`` (base frame): the target's motion and the offset's, added."""
    points, velocities, accelerations, _ = target_motion(target, times)
    shift, shift_velocity, shift_acceleration = offset(np.asarray(times, float))
    return ToolPath(np.asarray(times, float), points + shift, velocities + shift_velocity, accelerations + shift_acceleration,
                    np.asarray(rotation, float))


def coast(point: Any, velocity: Any, steps: int, rate_hz: float, rotation: np.ndarray, *, begin_s: float = 0.0,
          displacement: Any = (0.0, 0.0, 0.0), end_point: Any = None, stride: int = 1) -> ToolPath:
    """From ``point`` moving at ``velocity`` (zero acceleration) to rest ``steps`` samples later:

        p(tau) = p0 + V T (tau - S(tau)) + D s(tau)    (s minimum-jerk, S its integral; T the duration)

    so the velocity is V (1 - s) + D s'/T: V at the start, 0 at the end, and the acceleration 0 at both. The braking alone
    carries the tool V T / 2 further along the way it was going; ``displacement`` D is travelled on top (a leave along a
    direction), or ``end_point`` fixes where it stops (D = end - p0 - V T / 2). ``stride``: every stride-th sample and the last."""
    point, velocity = np.asarray(point, float), np.asarray(velocity, float)
    duration = steps / rate_hz
    rows = np.unique(np.append(np.arange(0, steps + 1, stride), steps))
    tau = rows / steps
    change = np.asarray(displacement, float) if end_point is None else np.asarray(end_point, float) - point - 0.5 * velocity * duration
    s, ds, dds, integral = minimum_jerk(tau)
    points = point + velocity * duration * (tau - integral)[:, None] + change * s[:, None]
    velocities = velocity * (1.0 - s)[:, None] + change * ds[:, None] / duration
    accelerations = -velocity * ds[:, None] / duration + change * dds[:, None] / duration**2
    return ToolPath(begin_s + rows / rate_hz, points, velocities, accelerations, np.asarray(rotation, float))


# ====================================================================== joint motion that gives the tool's


def jacobian(space: JointSpace, q: Any) -> np.ndarray:
    """(6, J) linear over angular, or (3, J) for a position-only kinematics: central differences of ``space.forward``."""
    q = np.asarray(q, float)
    rows = 3 if space.position_only else 6
    out = np.empty((rows, space.dof))
    for joint in range(space.dof):
        step = np.zeros(space.dof)
        step[joint] = JACOBIAN_STEP_RAD
        ahead, behind = space.forward(q + step), space.forward(q - step)
        out[:3, joint] = (ahead[:3, 3] - behind[:3, 3]) / (2 * JACOBIAN_STEP_RAD)
        if rows == 6:
            turn = ahead[:3, :3] @ behind[:3, :3].T
            # WHY the skew part alone: over 2e-6 rad it is the rotation vector to 1e-12 (sin x = x - x^3/6)
            out[3:, joint] = 0.5 * np.array([turn[2, 1] - turn[1, 2], turn[0, 2] - turn[2, 0], turn[1, 0] - turn[0, 1]]) / (2 * JACOBIAN_STEP_RAD)
    return out


def joint_motion(space: JointSpace, q: Any, velocity: Any, acceleration: Any) -> tuple[np.ndarray, np.ndarray]:
    """The joint velocity and acceleration that give the tool ``velocity`` and ``acceleration`` (linear, no turning) at ``q``:
    qd = J^+ v, qdd = J^+ (a - Jdot qdot), Jdot qdot by central differences of J along qdot."""
    q = np.asarray(q, float)
    matrix = jacobian(space, q)
    twist, push = np.zeros(matrix.shape[0]), np.zeros(matrix.shape[0])
    twist[:3], push[:3] = velocity, acceleration
    qd = np.linalg.lstsq(matrix, twist, rcond=None)[0]
    speed = float(np.max(np.abs(qd)))
    if speed > 0:
        delta = CURVATURE_STEP_RAD / speed
        push = push - (jacobian(space, q + delta * qd) - jacobian(space, q - delta * qd)) @ qd / (2 * delta)
    return qd, np.linalg.lstsq(matrix, push, rcond=None)[0]


def _miss(space: JointSpace, pose: np.ndarray, q: np.ndarray | None, tolerance_m: float, tolerance_rad: float) -> str:
    """Why an inverse-kinematics answer does not serve ``pose`` ("" when it does): none, outside the limits, or off the pose."""
    if q is None:
        return "the kinematics found no inverse-kinematics solution"
    bounds = space.bounds
    outside = np.flatnonzero((q < bounds.low - 1e-9) | (q > bounds.high + 1e-9))
    if len(outside):
        joint = int(outside[0])
        return (f"the solution puts {space.joint_names[joint]} at {math.degrees(q[joint]):.2f}°, outside its limits "
                f"{math.degrees(bounds.low[joint]):.2f}..{math.degrees(bounds.high[joint]):.2f}°")
    reached = space.forward(q)
    miss = float(np.linalg.norm(reached[:3, 3] - pose[:3, 3]))
    if miss > tolerance_m:
        return f"the solution misses the point by {1000 * miss:.3f} mm (tolerance {1000 * tolerance_m:.3f} mm)"
    if not space.position_only and rotation_angle(reached[:3, :3], pose[:3, :3]) > tolerance_rad:
        return f"the solution misses the orientation by {math.degrees(rotation_angle(reached[:3, :3], pose[:3, :3])):.3f}°"
    return ""


Refuse = Callable[[int, np.ndarray, str], None]


def out_of_reach(label: str) -> Refuse:
    """A refusal for a sample on the moving target: ``CannotCatch("intercept_out_of_reach")``, saying where and why."""

    def refuse(row: int, point: np.ndarray, why: str) -> None:
        where = ", ".join(f"{1000 * value:.1f}" for value in point)
        raise CannotCatch("intercept_out_of_reach", f"{label}: the target takes the tool to [{where}] mm: {why}")

    return refuse


def follow_tool(space: JointSpace, path: ToolPath, seed: Any, *, refuse: Refuse, first: State | None = None,
                last: State | None = None, pace: str | None = None, tolerance_m: float = 1e-4, tolerance_rad: float = 1e-3,
                stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint positions, velocities and accelerations along ``path``: inverse kinematics at every sample (each seeded from the
    last), ``joint_motion`` for the derivatives. ``first`` / ``last``: states the path must begin / end in exactly (a boundary
    handed on), put in as they are. ``pace``: a label; then a sample that needs more than a joint's velocity or acceleration
    limit raises ``CannotCatch("target_outruns_arm")`` at once, before the target can leave the reach further on.
    ``refuse(row, point, why)`` raises for a sample the kinematics cannot serve."""
    count = len(path.times)
    q, qd, qdd = (np.empty((count, space.dof)) for _ in range(3))
    here = np.asarray(seed, float)
    for row in range(count):
        if row == 0 and first is not None:
            q[0], qd[0], qdd[0] = first.q, first.qd, first.qdd
        elif row == count - 1 and last is not None:
            q[row], qd[row], qdd[row] = last.q, last.qd, last.qdd
        else:
            pose = path.pose(row)
            solution = space.inverse(pose, here)
            why = _miss(space, pose, solution, tolerance_m, tolerance_rad)
            if why:
                refuse(row, path.points[row], why)
            if row > 0:
                step = np.abs(solution - here)
                if np.max(step) > MAX_JOINT_STEP_RAD * stride:
                    joint = int(np.argmax(step))
                    refuse(row, path.points[row], f"{space.joint_names[joint]} jumps {math.degrees(step[joint]):.2f}° between samples: "
                                                  "a branch flip or a singularity")
            q[row] = solution
            qd[row], qdd[row] = joint_motion(space, solution, path.velocities[row], path.accelerations[row])
        here = q[row]
        if pace is not None:
            _keep_pace(space, pace, float(path.times[row]), qd[row], qdd[row])
    return q, qd, qdd


def _keep_pace(space: JointSpace, label: str, time_s: float, qd: np.ndarray, qdd: np.ndarray) -> None:
    bounds = space.bounds
    for quantity, values, limit, unit in (("velocity", qd, bounds.max_velocity, "°/s"), ("acceleration", qdd, bounds.max_acceleration, "°/s²")):
        ratio = np.abs(values) / limit
        joint = int(np.argmax(ratio))
        if ratio[joint] > 1 + 1e-6:
            # WHY bam_traj_gen's evidence (traj_generator.py _tracking_demand): the joint, what keeping pace needs, its limit
            raise CannotCatch("target_outruns_arm", f"{label}: keeping pace at t = {time_s:.3f} s needs {space.joint_names[joint]} at "
                                                    f"{math.degrees(abs(values[joint])):.1f}{unit}, {ratio[joint]:.2f}x its {quantity} limit "
                                                    f"{math.degrees(limit[joint]):.1f}{unit}")


def worst(space: JointSpace, qd: np.ndarray, qdd: np.ndarray) -> tuple[float, float]:
    """(max |qd| / velocity limit, max |qdd| / acceleration limit) over every sample and joint."""
    bounds = space.bounds
    return float(np.max(np.abs(qd) / bounds.max_velocity)), float(np.max(np.abs(qdd) / bounds.max_acceleration))


def piece(path: ToolPath, motion: tuple[np.ndarray, np.ndarray, np.ndarray], label: str) -> Piece:
    """A kernel piece from a followed path (t from 0), its tool points in the ``tool_target`` channel."""
    q, qd, qdd = motion
    return Piece(path.times - path.times[0], q, qd, qdd, label, channels={TOOL_TARGET: path.points.copy()})


# ====================================================================== the three motions every lane-F kind is made of


def synced(space: JointSpace, path: ToolPath, seed: Any, *, label: str, first: State | None = None,
           tolerance_m: float = 1e-4, tolerance_rad: float = 1e-3) -> Piece:
    """The tool kept on a moving target along ``path``, every sample: ``CannotCatch("target_outruns_arm")`` when a joint
    cannot keep pace, ``CannotCatch("intercept_out_of_reach")`` when the target takes the tool where the kinematics cannot."""

    def refuse(row: int, point: np.ndarray, why: str) -> None:
        where = ", ".join(f"{1000 * value:.1f}" for value in point)
        raise CannotCatch("intercept_out_of_reach", f"{label}: at t = {path.times[row]:.3f} s the target takes the tool to "
                                                    f"[{where}] mm: {why}")

    return piece(path, follow_tool(space, path, seed, refuse=refuse, first=first, pace=label, tolerance_m=tolerance_m,
                                   tolerance_rad=tolerance_rad), label)


def fitted(space: JointSpace, make: Callable[[int, int], ToolPath], steps: int, seed: Any, *, label: str, first: State | None = None,
           last: State | None = None, refuse: Refuse | None = None, tolerance_m: float = 1e-4, tolerance_rad: float = 1e-3,
           max_steps: int | None = None) -> Piece:
    """The piece ``make(steps, stride)`` (a ToolPath over ``steps`` samples) at the fewest steps from ``steps`` up that keep
    every joint inside its velocity and acceleration limits: screened at every ``FIT_STRIDE``-th sample, then checked at
    every sample. ``max_steps``: more than this is a refusal (a fixed time to fit into)."""

    steps = max(2, int(steps))

    def unreachable(row: int, point: np.ndarray, why: str) -> None:
        raise Unreachable(label, index=row, count=steps + 1, point=point, why=why)

    refuse = refuse or unreachable
    tried = []
    for _ in range(FIT_TRIES):
        if max_steps is not None and steps > max_steps:
            break
        screen = make(steps, FIT_STRIDE)
        _, qd, qdd = follow_tool(space, screen, seed, refuse=refuse, first=first, last=last, tolerance_m=tolerance_m,
                                 tolerance_rad=tolerance_rad, stride=FIT_STRIDE)
        velocity, acceleration = worst(space, qd, qdd)
        if velocity <= FIT_MARGIN and acceleration <= FIT_MARGIN:
            path = make(steps, 1)
            motion = follow_tool(space, path, seed, refuse=refuse, first=first, last=last, tolerance_m=tolerance_m,
                                 tolerance_rad=tolerance_rad)
            velocity, acceleration = worst(space, motion[1], motion[2])
            if velocity <= 1 - 1e-9 and acceleration <= 1 - 1e-9:
                return piece(path, motion, label)
        tried.append((steps, velocity, acceleration))
        steps = math.ceil(steps * max(1.05, min(2.0, 1.02 * max(velocity, math.sqrt(acceleration)))))
    last_try = tried[-1] if tried else (steps, math.nan, math.nan)
    raise ValueError(f"{label}: no duration {'up to ' + format(max_steps / space.rate_hz, '.3f') + ' s ' if max_steps else ''}keeps "
                     f"every joint inside its limits (at {last_try[0] / space.rate_hz:.3f} s: {last_try[1]:.2f}x the velocity limit, "
                     f"{last_try[2]:.2f}x the acceleration limit)")


def hold_steps(q: Any, steps: int, rate_hz: float, label: str) -> Piece:
    """At rest at ``q`` for exactly ``steps`` samples (kernels.hold rounds a duration; a sync time must land exactly)."""
    q = np.asarray(q, float)
    count = steps + 1
    zeros = np.zeros((count, q.size))
    return Piece(np.arange(count) / rate_hz, np.tile(q, (count, 1)), zeros, zeros.copy(), label,
                 channels={TOOL_TARGET: np.full((count, 3), np.nan)})


def approach(space: JointSpace, start: Any, arrive: State, steps: int, law: WaypointTrajectoryGenerator, *, label: str, wait_label: str,
             point: Any, what: str) -> tuple[list[Piece], list[str]]:
    """From rest at ``start`` to the moving state ``arrive`` exactly ``steps`` samples later, in joint space.

    The leg is the waypoint trajectory generator's (the quintic through both states: ``WaypointTrajectoryGenerator.leg(..., end_state=)``). WHY wait at rest first when
    it is shorter, rather than stretch it over the time there is: a quintic stretched to a moving end backs up first (its cubic
    term turns negative once the leg is longer than about 2.5 x distance / end speed). When the law's own leg is too long, the
    fallback is minimum-jerk over exactly the time there is, at the limits (noted); when not even that fits:
    ``CannotCatch("time_law_error")`` (bam_traj_gen's ``ruckig_error``: the intercept leg cannot be timed)."""
    if law.shape == "trapezoid":
        raise ValueError(f"{label}: a trapezoid is rest-to-rest; the approach ends moving with the target, so it takes "
                         "WaypointTrajectoryGenerator.minimum_jerk (the quintic through both states) or a backend that serves boundary "
                         "states")
    rate = space.rate_hz
    if steps < 1:
        raise CannotCatch("time_law_error", f"{what}: the moving state is due at t = {steps / rate:.3f} s, leaving no time to approach it")
    end = BoundaryState(np.asarray(arrive.qd, float), np.asarray(arrive.qdd, float))
    notes: list[str] = []
    try:
        leg = law.leg(start, arrive.q, space.limits, rate, label=label, end_state=end, dynamics=space.dynamics)
    except ValueError:
        leg = None
    if leg is None or len(leg.t) - 1 > steps:
        at_limits = WaypointTrajectoryGenerator.minimum_jerk(move_s=steps / rate)
        try:
            leg = at_limits.leg(start, arrive.q, space.limits, rate, label=label, end_state=end)
        except ValueError as error:
            raise CannotCatch("time_law_error", f"{what}: {error}") from error
        if len(leg.t) - 1 != steps:
            raise CannotCatch("time_law_error", f"{what}: the approach needs {(len(leg.t) - 1) / rate:.3f} s inside the limits, "
                                                f"but the target is met at t = {steps / rate:.3f} s")
        notes.append(f"{what}: the approach runs at the limits (minimum-jerk over {steps / rate:.3f} s) to meet the target in time")
    bounds = space.bounds
    beyond = np.maximum(np.max(bounds.low - leg.q, axis=0), np.max(leg.q - bounds.high, axis=0))
    if np.any(beyond > 1e-9):
        joint = int(np.argmax(beyond))
        raise CannotCatch("time_law_error", f"{what}: the approach to the moving state overshoots {space.joint_names[joint]}'s range by "
                                            f"{math.degrees(beyond[joint]):.2f}°")
    for row in range(0, len(leg.t), 5):
        try:
            space.forward(leg.q[row])
        except ValueError as error:
            raise CannotCatch("time_law_error", f"{what}: the approach passes where the kinematics has no tool pose ({error})") from error
    wait = steps - (len(leg.t) - 1)
    pieces: list[Piece] = []
    if wait > 0:
        pieces.append(hold_steps(start, wait, rate, wait_label))
        notes.append(f"{what}: waits {wait / rate:.3f} s at the start, then approaches")
    target = np.full((len(leg.t), 3), np.nan)
    target[-1] = point  # WHY only the last row: the approach's tool path is not commanded, only where it meets the target
    pieces.append(Piece(leg.t, leg.q, leg.qd, leg.qdd, label, channels={TOOL_TARGET: target}))
    return pieces, notes


def meet(space: JointSpace, start: Any, moving: Any, first: int, last: int, *, rotation: np.ndarray, offset: Any,
         waypoint_generator: WaypointTrajectoryGenerator, label: str, tolerance_m: float = 1e-4) -> tuple[list[Piece], list[str],
                                                                                                      ToolPath]:
    """From rest at ``start``: the approach, then the tool on the target (plus a fixed ``offset``, base frame) from sample
    ``first`` to ``last``, velocity matched at ``first``. The pieces (the last is the synchronised one), notes, and its path."""
    rate = space.rate_hz
    t0, t1 = first / rate, last / rate
    check_window(moving, t0, t1, "the sync window")
    path = chase(moving, np.arange(first, last + 1) / rate, rotation, fixed_offset(offset))
    seed = solve(space, path.pose(0), start, what=f"{label}: the tool on the target at t = {t0:.3f} s", tolerance_m=tolerance_m)
    track = synced(space, path, seed, label=f"{label}: synchronised", tolerance_m=tolerance_m)
    pieces, notes = approach(space, start, State(track.q[0], track.qd[0], track.qdd[0]), first, waypoint_generator,
                             label=f"{label}: approach", wait_label=f"{label}: wait at the start", point=path.points[0],
                             what=label)
    return [*pieces, track], notes, path


def solve(space: JointSpace, pose: np.ndarray, seed: Any, *, what: str, tolerance_m: float = 1e-4,
          tolerance_rad: float = 1e-3, cannot_catch: bool = True) -> np.ndarray:
    """Inverse kinematics at one pose (seeded), refused by name: ``intercept_out_of_reach`` for a pose on the target, else
    ``Unreachable``."""
    solution = space.inverse(pose, seed)
    why = _miss(space, pose, solution, tolerance_m, tolerance_rad)
    if why:
        if cannot_catch:
            raise CannotCatch("intercept_out_of_reach", f"{what}: {why}")
        raise Unreachable(what, index=0, count=1, point=pose[:3, 3], why=why)
    return solution


def law_steps(space: JointSpace, law: WaypointTrajectoryGenerator, q0: Any, q1: Any) -> int:
    """The samples a rest-to-rest leg q0 -> q1 takes under ``law`` (a duration to start a fit from)."""
    if law.built_in:
        return math.ceil(law.leg_timing(np.asarray(q0, float), np.asarray(q1, float), space.limits).duration_s * space.rate_hz - 1e-9)
    return len(law.leg(q0, q1, space.limits, space.rate_hz, dynamics=space.dynamics).t) - 1


def braking_steps(space: JointSpace, qd: Any) -> int:
    """Samples to brake from ``qd`` at about half the acceleration limit. WHY 3 v / a: a minimum-jerk decay of the velocity
    peaks at 1.875 v / T, so T = 3 v / a keeps it near 60 % of the limit; ``fitted`` then lengthens it if the path needs."""
    return math.ceil(3.0 * float(np.max(np.abs(np.asarray(qd, float)) / space.bounds.max_acceleration)) * space.rate_hz)


__all__: list[str] = []  # WHY empty: a private helper; the kinds import its names directly
