from __future__ import annotations

import math

import numpy as np

from ..cartesian import Unreachable, rotation_angle, solve_points
from ..registry import LIMITS, kind
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator
from ._lane_d import pose_of, pose_set, rotation_about, rotation_between, try_solve, usable


def rotation_axes_spread(rotations: list[np.ndarray]) -> float:
    """sigma_3 / sigma_1 of the consecutive relative rotation axes (0 when every rotation axis is parallel, which AX = XB cannot
    solve for the hand-eye rotation; larger is better)."""
    axes = []
    for before, after in zip(rotations[:-1], rotations[1:], strict=True):
        relative = before.T @ after
        vector = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]])
        if np.linalg.norm(vector) > 1e-9:
            axes.append(vector / np.linalg.norm(vector))
    if len(axes) < 3:
        return 0.0
    sigma = np.linalg.svd(np.array(axes), compute_uv=False)
    return float(sigma[2] / sigma[0])


@kind(use_cases=("hand_eye_calibration", "accuracy"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model", "reach_map"), catalog_id="hand_eye_station_set")
def hand_eye_station_set(space, *, stations=30, min_rotation_between_rad=0.52, check_set_stations=15, standoff_m=0.25,
                         cone_rad=0.8, roll_rad=1.2, focus_m=None, candidates=800, dwell_s=1.0,
                         waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5), seed=0,
                         label="hand-eye stations", start=None):
    """Tool poses around one point for AX = XB hand-eye calibration, then a held-out check set (a pose set: ``stations`` +
    ``check_set_stations`` stations, labelled ``"<label>: station k"``; the entry's ``check_set`` lists the check ones).

    Every station aims the tool's +z axis at ``focus_m`` (default: the point ``standoff_m`` ahead of the tool along its start +z)
    from ``standoff_m`` (varied 0.8-1.2x), inside a cone of half-angle ``cone_rad`` about the start's aim and rolled up to
    ``roll_rad`` about it. The stations are chosen from ``candidates`` seeded candidates that solve (``solve_points`` is run on the
    chosen ones) greedily: each next one is the candidate farthest in rotation from every one chosen so far, among those whose
    rotation from the last is at least ``min_rotation_between_rad`` (AX = XB uses the motions between consecutive stations: Tsai
    & Lenz, Park & Martin). If the candidates cannot supply that many such stations the call is refused saying how many it found.
    The entry records the smallest consecutive rotation and the spread of the rotation axes (sigma_3 / sigma_1; 0 = all parallel).

    Task-agnostic: a point to aim at and a spread of orientations; a camera on the tool or a marker on it are the caller's."""
    total = stations + check_set_stations
    if stations < 3 or check_set_stations < 0:
        raise ValueError("a hand-eye set needs at least 3 stations (and a check set of 0 or more)")
    rng = np.random.default_rng(seed)
    room = space.room(start)
    here = space.forward(start)
    rotation0, aim = here[:3, :3], here[:3, :3][:, 2]
    focus = here[:3, 3] + standoff_m * aim if focus_m is None else np.asarray(focus_m, float)
    pool = []  # (pose, q)
    for _ in range(candidates):
        polar = cone_rad * math.sqrt(rng.uniform())
        azimuth = rng.uniform(0.0, 2.0 * math.pi)
        side = np.cross(aim, [0.0, 0.0, 1.0] if abs(aim[2]) < 0.9 else [1.0, 0.0, 0.0])
        direction = rotation_about(aim, azimuth) @ rotation_about(side / np.linalg.norm(side), polar) @ aim
        rotation = rotation_about(direction, rng.uniform(-roll_rad, roll_rad)) @ rotation_between(aim, direction) @ rotation0
        pose = pose_of(rotation, focus - standoff_m * rng.uniform(0.8, 1.2) * direction)
        q = try_solve(space, pose, start, room)
        if q is not None:
            pool.append((pose, q))
    chosen: list[int] = []
    before = rotation0
    spread_to = [rotation0]
    while len(chosen) < total:
        best, best_score = None, -1.0
        for index, (pose, _) in enumerate(pool):
            if index in chosen:
                continue
            turn = rotation_angle(before, pose[:3, :3])
            if chosen and turn < min_rotation_between_rad:
                continue
            score = min(rotation_angle(other, pose[:3, :3]) for other in spread_to)
            if score > best_score:
                best, best_score = index, score
        if best is None:
            found = len(chosen)
            raise ValueError(f"{label}: found {found} of {total} stations with at least {math.degrees(min_rotation_between_rad):.1f}° "
                             f"between consecutive ones among {len(pool)} reachable of {candidates} candidates; widen cone_rad "
                             f"({math.degrees(cone_rad):.0f}°) or roll_rad ({math.degrees(roll_rad):.0f}°), or ask for fewer")
        chosen.append(best)
        before = pool[best][0][:3, :3]
        spread_to.append(before)
    poses = [pool[index][0] for index in chosen]
    solved = solve_points(space, start, poses, what=label)
    # WHY the screened answer where the chained one is not usable: solve_points checks the limits, not the room or a collision
    solutions = [q if usable(space, q, room) else pool[index][1] for q, index in zip(solved, chosen, strict=True)]
    turns = [rotation_angle(a[:3, :3], b[:3, :3]) for a, b in zip(poses[:-1], poses[1:], strict=True)]
    spread = rotation_axes_spread([p[:3, :3] for p in poses])
    points = [pose[:3, 3] for pose in poses]
    return pose_set(space, label, start, solutions, points=points, dwell_s=dwell_s,
                    waypoint_generator=waypoint_generator, room=room, via=start,
                    notes=[f"smallest rotation between consecutive stations {math.degrees(min(turns)):.1f}°; rotation-axis spread "
                           f"(sigma_3 / sigma_1) {spread:.3f}"],
                    entry={"focus_m": [float(v) for v in focus], "check_set": list(range(stations + 1, total + 1)),
                           "min_consecutive_rotation_rad": float(min(turns)), "rotation_axis_spread": spread})
