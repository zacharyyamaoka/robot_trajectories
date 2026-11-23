from __future__ import annotations

import math

import numpy as np

from ..cartesian import Unreachable
from ..registry import LIMITS, kind
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator
from ._lane_d import pose_of, pose_set, rotation_about, rotation_between, try_solve, unit

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


@kind(use_cases=("camera_calibration",), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model", "reach_map"), catalog_id="camera_board_views")
def camera_board_views(space, *, views=20, max_tilt_rad=0.785, image_cells=9, standoff_m=0.3, half_field_rad=0.35, focus_m=None,
                       dwell_s=1.0, waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5),
                       label="board views", start=None):
    """Tool poses that show one flat target (a printed board, at ``focus_m``) to a camera on the tool at many tilts and positions
    across the image: Zhang's views for intrinsics and distortion (a pose set: ``views`` stations, ``"<label>: station k"``).

    The tool's +z axis is the optical axis, the board's normal faces the tool at its start (the focus defaults to the point
    ``standoff_m`` ahead of the tool along its start +z). View j puts the board's centre in image cell ``j mod image_cells`` of
    a square grid of cells (a full-image offset of ``half_field_rad``, so the board visits the corners and the middle), at one of
    three depths (0.8, 1.0, 1.2 x ``standoff_m``), the board turned away from the optical axis by a tilt that grows with j up
    to ``max_tilt_rad`` and sweeps round it by the golden angle (so tilts fall in every direction). A candidate whose inverse
    kinematics has no usable answer is skipped (the notes count them); the call is refused if ``views`` cannot be made from
    ``6 x views`` candidates. Deterministic: there is nothing to seed."""
    side = round(math.sqrt(image_cells))
    if side * side != image_cells or views < 1:
        raise ValueError(f"image_cells must be a square (4, 9, 16...) and views at least 1, got {image_cells} and {views}")
    room = space.room(start)
    here = space.forward(start)
    rotation0, aim = here[:3, :3], here[:3, :3][:, 2]
    focus = here[:3, 3] + standoff_m * aim if focus_m is None else np.asarray(focus_m, float)
    normal = -aim  # the board faces the tool at its start
    # WHY a serpentine over the cells: consecutive views are neighbours in the image, so the joint legs between them stay short
    cells = [(column - (side - 1) / 2.0, row - (side - 1) / 2.0) for row in range(side)
             for column in (range(side) if row % 2 == 0 else reversed(range(side)))]
    poses, solutions, seed_q, skipped = [], [], np.asarray(start, float), 0
    for candidate in range(6 * views):
        if len(poses) == views:
            break
        cell = cells[candidate % image_cells]
        offset = np.array([cell[0], cell[1]]) / max(1.0, (side - 1) / 2.0) * half_field_rad
        tilt = max_tilt_rad * (0.3 + 0.7 * ((candidate * 0.381966) % 1.0))
        around = candidate * GOLDEN_ANGLE
        e1 = unit(np.cross(normal, [0.0, 0.0, 1.0] if abs(normal[2]) < 0.9 else [1.0, 0.0, 0.0]))
        e2 = np.cross(normal, e1)
        toward_tool = math.cos(tilt) * normal + math.sin(tilt) * (math.cos(around) * e1 + math.sin(around) * e2)
        look = -toward_tool  # from the tool to the board
        depth = standoff_m * (0.8, 1.0, 1.2)[candidate % 3]
        # the camera frame: z along ``look`` (twist-free from the start's), then turned so the board's centre appears at ``offset``
        straight = rotation_between(aim, look) @ rotation0
        in_camera = unit([math.tan(offset[0]), math.tan(offset[1]), 1.0])
        rotation = straight @ rotation_between(in_camera, [0.0, 0.0, 1.0])
        position = focus - depth * (rotation @ in_camera)
        pose = pose_of(rotation, position)
        q = try_solve(space, pose, seed_q, room)
        if q is None:
            skipped += 1
            continue
        poses.append(pose)
        solutions.append(q)
        seed_q = q
    if len(poses) < views:
        raise Unreachable(label, index=len(poses), count=views, point=focus,
                          why=f"only {len(poses)} of the {views} views have a usable inverse-kinematics answer among {6 * views} candidates")
    notes = [f"{skipped} candidate views skipped (no usable inverse-kinematics answer)"] if skipped else []
    return pose_set(space, label, start, solutions, points=[pose[:3, 3] for pose in poses], dwell_s=dwell_s,
                    waypoint_generator=waypoint_generator, room=room, via=start, notes=notes,
                    entry={"focus_m": [float(v) for v in focus], "image_cells": int(image_cells), "max_tilt_rad": float(max_tilt_rad)})
