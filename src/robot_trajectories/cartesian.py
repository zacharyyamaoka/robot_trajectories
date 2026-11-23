"""Tool paths to joint trajectories, with the kinematics passed in: the seam every Cartesian kind is built on.

Zach (2026-11-18): "the U shaped pick and place shape is also just a robot trajectory... it's just that now we are like
putting in cartesian points... those require FK/IK that you need to pass in, but the output is the same, just joint
trajectories". So nothing here knows a robot: the space's ``forward`` / ``inverse`` do, and what comes out is joint motion.

- ``follow(space, start, path, law, ...)``: the tool along a geometric path (a function of progress s in [0, 1] returning
  4x4 poses), inverse kinematics at every step, as MoveIt's ``computeCartesianPath`` does (an end-effector step plus a jump
  threshold), then timed by a WaypointTrajectoryGenerator over the joint path. Refused, never clipped, at the first sample that fails, saying
  which sample, where the tool was and why.
- ``line(a, b)``: the straight path between two poses (position linear, orientation slerped).
- ``solve_points(space, start, poses, ...)``: inverse kinematics at a few points only, each seeded from the last.
- ``poses_for(space, start, points_or_poses, orientation)``: points or poses as 4x4 targets ("hold" the start's orientation,
  or "follow" the poses given).

A path's commanded tool point at every sample goes in the piece's ``tool_target`` channel (NaN where none is commanded), so
a caller holds the output to it with forward kinematics (tests/test_cartesian.py: within 1 mm).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from robot_msgs.geometry.rotations import quat_from_matrix, quat_slerp, quat_to_matrix

from .segments import TOOL_TARGET, Piece
from .space import JointSpace
from .waypoint_trajectory_generator import WaypointTrajectoryGenerator

ORIENTATIONS = ("hold", "follow")
Path = Callable[[np.ndarray], np.ndarray]  # progress s (M,) in [0, 1] -> poses (M, 4, 4)


class Unreachable(ValueError):
    """A tool path or point the kinematics cannot follow: which sample, where the tool was, and why."""

    def __init__(self, what: str, *, index: int, count: int, point: np.ndarray, why: str):
        self.index, self.count, self.point, self.why = index, count, np.asarray(point, float), why
        where = ", ".join(f"{1000 * value:.1f}" for value in self.point)
        super().__init__(f"{what}: sample {index} of {count} (tool at [{where}] mm) is unreachable: {why}")


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """The angle (rad) of the rotation between two 3x3 rotations."""
    cosine = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, cosine))))


def poses_for(space: JointSpace, start: np.ndarray, points_or_poses: Sequence[Any], orientation: str = "hold") -> list[np.ndarray]:
    """Targets as 4x4 poses. Points (xyz, m) take the orientation the tool has at ``start`` ("hold"); poses keep their own
    ("follow") or take the start's too ("hold")."""
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation is one of {ORIENTATIONS}, got {orientation!r}")
    held = space.forward(start)[:3, :3]
    out = []
    for index, item in enumerate(points_or_poses):
        array = np.asarray(item, float)
        if array.shape == (3,):
            if orientation == "follow":
                raise ValueError(f"point {index} is an xyz: orientation='follow' needs 4x4 poses (or use 'hold')")
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = held, array
        elif array.shape == (4, 4):
            pose = array.copy()
            if orientation == "hold":
                pose[:3, :3] = held
        else:
            raise ValueError(f"target {index} has shape {array.shape}: give xyz points (3,) or 4x4 poses")
        out.append(pose)
    return out


def line(a: np.ndarray, b: np.ndarray) -> Path:
    """The straight tool path from pose ``a`` to pose ``b``: position linear in s, orientation slerped."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    qa, qb = quat_from_matrix(a[:3, :3]), quat_from_matrix(b[:3, :3])

    def at(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, float)
        poses = np.tile(np.eye(4), (len(s), 1, 1))
        poses[:, :3, 3] = a[:3, 3] + (b[:3, 3] - a[:3, 3]) * s[:, None]
        poses[:, :3, :3] = quat_to_matrix(quat_slerp(qa, qb, s))
        return poses

    return at


def _check(space: JointSpace, what: str, index: int, count: int, target: np.ndarray, q: np.ndarray | None,
           tolerance_m: float, tolerance_rad: float) -> np.ndarray:
    """The IK answer at one sample, or Unreachable naming why."""
    point = target[:3, 3]
    if q is None:
        raise Unreachable(what, index=index, count=count, point=point, why="the kinematics found no inverse-kinematics solution")
    bounds = space.bounds
    outside = np.flatnonzero((q < bounds.low - 1e-9) | (q > bounds.high + 1e-9))
    if len(outside):
        joint = int(outside[0])
        raise Unreachable(what, index=index, count=count, point=point,
                          why=f"the solution puts {space.joint_names[joint]} at {math.degrees(q[joint]):.2f}°, outside its limits "
                              f"{math.degrees(bounds.low[joint]):.2f}..{math.degrees(bounds.high[joint]):.2f}°")
    reached = space.forward(q)
    miss = float(np.linalg.norm(reached[:3, 3] - point))
    if miss > tolerance_m:
        raise Unreachable(what, index=index, count=count, point=point,
                          why=f"the solution misses the point by {1000 * miss:.3f} mm (tolerance {1000 * tolerance_m:.3f} mm)")
    if not space.position_only:
        turn = rotation_angle(reached[:3, :3], target[:3, :3])
        if turn > tolerance_rad:
            raise Unreachable(what, index=index, count=count, point=point,
                              why=f"the solution misses the orientation by {math.degrees(turn):.3f}°")
    return q


def solve_points(space: JointSpace, start: np.ndarray, poses: Sequence[np.ndarray], *, what: str = "points",
                 tolerance_m: float = 1e-4, tolerance_rad: float = 1e-3) -> list[np.ndarray]:
    """Joint positions at each pose, each solved seeded from the last (the first from ``start``)."""
    solutions, seed = [], np.asarray(start, float)
    for index, pose in enumerate(poses):
        q = _check(space, what, index, len(poses), pose, space.inverse(pose, seed), tolerance_m, tolerance_rad)
        solutions.append(q)
        seed = q
    return solutions


def follow(space: JointSpace, start: np.ndarray, path: Path, law: WaypointTrajectoryGenerator, *, length_m: float, label: str,
           eef_step_m: float = 0.001, jump_threshold: float = 10.0, max_joint_step_rad: float = 0.1,
           tolerance_m: float = 1e-4, tolerance_rad: float = 1e-3) -> Piece:
    """The tool along ``path`` (progress -> 4x4 poses) from the configuration ``start``, timed by ``law``, at rest at both ends.

    - ``eef_step_m``: the tool distance between inverse-kinematics samples (MoveIt's ``eef_step``; at least 50 samples).
    - ``jump_threshold``: a step in joint space larger than this many times the mean step is a jump (MoveIt's relative
      threshold: a branch flip or a singularity); ``max_joint_step_rad`` is the absolute one.
    - ``tolerance_m`` / ``tolerance_rad``: how far forward kinematics of a solution may land from its target.

    Orientation is checked only when the kinematics places a pose (not ``position_only``)."""
    start = np.asarray(start, float)
    count = max(51, int(math.ceil(length_m / eef_step_m)) + 1)
    grid = np.linspace(0.0, 1.0, count)
    targets = path(grid)
    here = space.forward(start)
    if np.linalg.norm(here[:3, 3] - targets[0][:3, 3]) > tolerance_m:
        raise Unreachable(label, index=0, count=count, point=targets[0][:3, 3],
                          why=f"the path does not begin where the tool is at the start configuration "
                              f"({1000 * np.linalg.norm(here[:3, 3] - targets[0][:3, 3]):.3f} mm away)")
    q = np.empty((count, space.dof))
    q[0] = start
    for index in range(1, count):
        solution = _check(space, label, index, count, targets[index], space.inverse(targets[index], q[index - 1]),
                          tolerance_m, tolerance_rad)
        step = np.abs(solution - q[index - 1])
        if np.max(step) > max_joint_step_rad:
            joint = int(np.argmax(step))
            raise Unreachable(label, index=index, count=count, point=targets[index][:3, 3],
                              why=f"{space.joint_names[joint]} jumps {math.degrees(step[joint]):.2f}° between samples "
                                  f"(max_joint_step_rad {math.degrees(max_joint_step_rad):.2f}°): a branch flip or a singularity")
        q[index] = solution
    steps = np.max(np.abs(np.diff(q, axis=0)), axis=1)
    mean = float(np.mean(steps))
    if mean > 0 and np.any(steps > jump_threshold * mean):
        index = int(np.argmax(steps > jump_threshold * mean)) + 1
        raise Unreachable(label, index=index, count=count, point=targets[index][:3, 3],
                          why=f"a joint-space jump {steps[index - 1] / mean:.1f}x the mean step (jump_threshold {jump_threshold:g})")
    # WHY the space's dynamics handed on: a waypoint trajectory generator whose backend needs them (TOPP-RA's torque mode) bounds torque along the path
    timed = law.along(q, space.limits, space.rate_hz, label=label, progress_grid=grid, dynamics=space.dynamics)
    arrays = timed.arrays
    if timed.progress is not None:
        tool = path(np.clip(timed.progress, 0.0, 1.0))[:, :3, 3]
    else:  # a backend that gives no progress: the tool point is commanded at the path's ends only
        tool = np.full((len(arrays.t), 3), np.nan)
        tool[0], tool[-1] = targets[0][:3, 3], targets[-1][:3, 3]
    return Piece(arrays.t, arrays.q, arrays.qd, arrays.qdd, label, channels={TOOL_TARGET: tool})


def tool_target_hold(point: np.ndarray, count: int) -> dict[str, np.ndarray]:
    """A ``tool_target`` channel that commands one point for ``count`` samples (a dwell at a Cartesian point)."""
    return {TOOL_TARGET: np.tile(np.asarray(point, float), (count, 1))}


__all__ = ["ORIENTATIONS", "Unreachable", "follow", "line", "poses_for", "rotation_angle", "solve_points", "tool_target_hold"]
