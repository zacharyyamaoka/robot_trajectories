from __future__ import annotations

import math

import numpy as np

from ..registry import LIMITS, kind
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator
from ._lane_d import line_clear, pose_of, pose_set, try_solve  # noqa: F401  (line_clear: the legs are routed by pose_set)

# WHY from memory, not re-read (ISO 9283 is paywalled; the catalog row says the same): the pose test is five poses on the
# diagonal plane of a cube, P1 at the plane's centre and P2-P5 at 10 % of the plane's diagonal in from its four corners, visited
# P1..P5 in the same order for 30 cycles, each pose approached from the same direction. A pose test of 100 %, 50 % and 10 % of
# the rated speed is the standard's too (speed_fraction). The numbers below are those, as remembered.
INSET_FRACTION = 0.1
EDGE_LADDER = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)  # fractions of the asked edge tried to name the largest that fits


def cube_poses(centre: np.ndarray, edge_m: float, azimuth_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """(P1..P5 (5, 3), the cube's eight corners (8, 3)) for a cube of ``edge_m`` about ``centre``, turned ``azimuth_rad`` about +z.

    The test plane holds two opposite vertical edges: corners (-h, -h, -+h) and (h, h, -+h); its diagonal is the cube's space
    diagonal, edge x sqrt(3). P2-P5 sit ``INSET_FRACTION`` of that diagonal in from the corners, towards the centre."""
    half = 0.5 * edge_m
    turn = np.array([[math.cos(azimuth_rad), -math.sin(azimuth_rad), 0.0], [math.sin(azimuth_rad), math.cos(azimuth_rad), 0.0],
                     [0.0, 0.0, 1.0]])
    plane = np.array([[-half, -half, -half], [half, half, -half], [half, half, half], [-half, -half, half]])
    toward_centre = -plane / np.linalg.norm(plane, axis=1, keepdims=True)
    diagonal = edge_m * math.sqrt(3.0)
    inner = plane + INSET_FRACTION * diagonal * toward_centre
    corners = np.array([[sx, sy, sz] for sx in (-half, half) for sy in (-half, half) for sz in (-half, half)])
    return np.vstack([np.zeros(3), inner]) @ turn.T + centre, corners @ turn.T + centre


def fit(space, rotation, q_seed, centre, edge_m, azimuth_rad, room):
    """The joint solutions at P1..P5 when the cube of ``edge_m`` about ``centre`` is wholly reachable (its five poses and its eight
    corners all solve inside the room, the tool keeping ``rotation``), else None. Every solve is seeded from P1's."""
    poses, corners = cube_poses(centre, edge_m, azimuth_rad)
    first = try_solve(space, pose_of(rotation, poses[0]), q_seed, room)
    if first is None:
        return None
    solutions = [first]
    for point in poses[1:]:
        q = try_solve(space, pose_of(rotation, point), first, room)
        if q is None or float(np.max(np.abs(q - first))) > 1.0:  # WHY 1 rad: a branch flip, not the same arm pose family
            return None
        solutions.append(q)
    for point in corners:
        if try_solve(space, pose_of(rotation, point), first, room) is None:
            return None
    return solutions


@kind(use_cases=("accuracy", "repeatability", "stop_accuracy"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model", "reach_map"), catalog_id="iso9283_pose_cube")
def iso9283_pose_cube(space, *, cube_edge_m=0.2, cycles=30, speed_fraction=1.0, dwell_s=2.0, centre_m=None, azimuth_rad=0.0,
                      label="pose test", start=None):
    """ISO 9283's pose test: five stations P1-P5 on the diagonal plane of a cube inside the reach, visited in order for ``cycles``
    cycles, each approached from the same direction, ``dwell_s`` at each (a pose set: stations joined by min-jerk legs).

    The tool keeps its start orientation, so ISO's position and orientation errors are scored at the same five stations every
    cycle. The cube (``cube_edge_m`` on a side, turned ``azimuth_rad`` about +z) is centred where the tool starts, or at
    ``centre_m`` (xyz, metres); with no centre given a few nearby ones are tried. It must lie wholly in the reach: its five poses
    and eight corners solve inside the room (``space.room``: a reach map keeps it reachable), else the call is refused naming
    the largest edge that fits where the tool starts. ``speed_fraction``: peak joint speed as a fraction of the limit (the
    standard tests 1.0, 0.5 and 0.1). The very first P1 is approached from the start, not from P5.

    Task-agnostic: five poses and a cube. A 3-D kind: a planar robot cannot reach a cube's corners and is refused."""
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    room = space.room(start)
    here = space.forward(start)
    rotation, point = here[:3, :3], here[:3, 3]
    if centre_m is not None:
        centres = [np.asarray(centre_m, float)]
    else:
        steps = np.array([[dx, dy, dz] for dx in (-1, -0.5, 0, 0.5, 1) for dy in (-1, -0.5, 0, 0.5, 1) for dz in (-0.5, 0, 0.5)])
        centres = sorted((point + cube_edge_m * offset for offset in steps), key=lambda candidate: float(np.linalg.norm(candidate - point)))
    solutions, used = None, None
    for candidate in centres:
        solutions = fit(space, rotation, start, candidate, cube_edge_m, azimuth_rad, room)
        if solutions is not None:
            used = candidate
            break
    if solutions is None:
        largest = next((cube_edge_m * fraction for fraction in EDGE_LADDER
                        if fit(space, rotation, start, centres[0], cube_edge_m * fraction, azimuth_rad, room) is not None), None)
        raise ValueError(f"{label}: a {cube_edge_m:.3f} m cube does not fit in the reach around the tool's start "
                         f"(the largest edge that does, on a ladder of fractions: "
                         f"{'none down to 20 %' if largest is None else f'{largest:.3f} m'}); a planar robot has no cube at all")
    poses, _ = cube_poses(used, cube_edge_m, azimuth_rad)
    stations = [solutions[index] for _ in range(cycles) for index in range(5)]
    points = [poses[index] for _ in range(cycles) for index in range(5)]
    law = WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=speed_fraction)
    return pose_set(space, label, start, stations, points=points, dwell_s=dwell_s, waypoint_generator=law, room=room,
                    via=start,
                    entry={"cube": {"centre_m": [float(v) for v in used], "edge_m": float(cube_edge_m), "azimuth_rad": float(azimuth_rad)},
                           "poses": ["P1", "P2", "P3", "P4", "P5"] * cycles, "cycles": int(cycles)})
