from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..cartesian import follow, line, poses_for, rotation_angle, tool_target_hold
from ..registry import LIMITS, kind
from ..segments import Piece
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator


@kind(use_cases=("tracking_eval", "accuracy", "workspace_coverage"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model",), catalog_id="cartesian_straight_line")
def cartesian_path(space, points_or_poses, *, waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5),
                   orientation="hold", dwell_s=0.0, eef_step_m=0.001, jump_threshold=10.0, max_joint_step_rad=0.1,
                   tolerance_m=1e-4, label="cartesian path", start=None):
    """The tool in straight lines from where it is through ``points_or_poses`` (xyz points in metres, or 4x4 poses, in the
    kinematics' base frame), inverse kinematics at every ``eef_step_m`` and refused at the first sample that fails (MoveIt's
    computeCartesianPath; Pilz's LIN). Each line is one rest-to-rest leg under ``waypoint_generator``; ``dwell_s`` holds at each point.

    ``orientation``: "hold" keeps the tool's orientation at the start; "follow" slerps to each pose's own. A position-only
    kinematics (a planar five-bar) ignores orientation. WHY a stop at each point: a corner turned at speed is an unbounded
    acceleration; a curve that does not stop is a smooth path kind (cartesian_arc, KINDS_TODO lane C)."""
    targets = poses_for(space, start, points_or_poses, orientation)
    if not targets:
        raise ValueError("cartesian_path needs at least one point or pose")
    pieces: list[k.Arrays] = []
    here_q, here_pose = start, space.forward(start)
    for index, target in enumerate(targets, start=1):
        length = float(np.linalg.norm(target[:3, 3] - here_pose[:3, 3]))
        turn = 0.0 if space.position_only else rotation_angle(here_pose[:3, :3], target[:3, :3])
        if length < 1e-9 and turn < 1e-9:
            continue
        # WHY 5 cm per radian: a pure turn still gets an IK sample every ~1 degree
        piece = follow(space, here_q, line(here_pose, target), waypoint_generator, length_m=max(length, 0.05 * turn),
                       label=f"{label}: line {index}", eef_step_m=eef_step_m, jump_threshold=jump_threshold,
                       max_joint_step_rad=max_joint_step_rad, tolerance_m=tolerance_m)
        pieces.append(piece)
        here_q, here_pose = piece.q[-1], target
        if dwell_s > 0:
            rest = k.hold(here_q, dwell_s, space.rate_hz, label=f"{label}: dwell")
            pieces.append(Piece(rest.t, rest.q, rest.qd, rest.qdd, rest.label, channels=tool_target_hold(target[:3, 3], len(rest.t))))
    if not pieces:
        raise ValueError("cartesian_path: every point is where the tool already is")
    return pieces
