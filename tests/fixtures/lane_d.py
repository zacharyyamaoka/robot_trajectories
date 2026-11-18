"""Lane D's fixtures (numpy only): the spaces its kinds are tested in, and the independent oracles that hold them.

Stand-ins, labelled: every robot is the round's fixture (``robots.py``: the five-bar's closed form, the UR5e's DH table); the READY
configurations below are made-up test values (a UR5e at its box's middle, the five-bar's bench home), not a rig's READY. The
oracles re-derive each standard's geometry from its definition rather than from the kind's code (an independent ruler)."""

from __future__ import annotations

import math

import numpy as np

from fixtures import robots
from robot_trajectories import JointSpace

FIVEBAR_START = np.radians([128.0, -126.0])  # well inside the reach map, away from the elbow singularity (KINDS_TODO)
FIVEBAR_READY = np.radians([100.0, -76.0])
UR5E_READY = np.radians([0.0, -90.0, 85.0, -95.0, -90.0, 0.0])  # the demo box's middle: elbow up, clear of the base
FIVEBAR_UP = (0.0, -1.0, 0.0)  # the plane frame's y-hat hangs downward (fb_bench/mapping.py): "up" is -y


def ur5e(*, named_home: bool = False) -> JointSpace:
    home = UR5E_READY if named_home else None
    space = JointSpace.from_description(robots.UR5E_DESCRIPTION, max_acceleration=6.0, home=home).within(robots.UR5E_DEMO_BOX)
    return space.with_kinematics(robots.UR5eKinematics())


def fivebar(*, reach_map: bool = False, named_home: bool = False) -> JointSpace:
    space = JointSpace.from_limits(robots.FIVEBAR_LIMITS, home=FIVEBAR_READY if named_home else np.radians([100.0, -76.0]))
    space = space.with_kinematics(robots.FiveBarKinematics())
    return space.with_reach_map(robots.fivebar_reach_map()) if reach_map else space


# ====================================================================== independent oracles


def pose_cube_points(centre: np.ndarray, edge_m: float) -> np.ndarray:
    """ISO 9283's P1-P5 from the definition, built a different way from the kind: P1 the cube's centre; P2-P5 the four corners of
    the diagonal-plane rectangle pulled to 0.8 of the way back to P1 (10 % of the plane's diagonal in from each corner, and the
    corner is half a diagonal from the centre). Rows P1..P5, a cube with the base frame's axes."""
    half = 0.5 * edge_m
    corners = np.array([[-half, -half, -half], [half, half, -half], [half, half, half], [-half, -half, half]])
    return np.vstack([centre, centre + 0.8 * corners])


def circle_error(points: np.ndarray, centre: np.ndarray, radius_m: float, normal: np.ndarray) -> float:
    """The worst distance (m) of ``points`` from the circle of ``radius_m`` about ``centre`` in the plane with ``normal``."""
    offsets = points - centre
    height = offsets @ normal
    in_plane = offsets - height[:, None] * normal
    return float(np.max(np.hypot(np.linalg.norm(in_plane, axis=1) - radius_m, height)))


def polyline_error(points: np.ndarray, vertices: list[np.ndarray]) -> float:
    """The worst distance (m) of ``points`` from the polyline through ``vertices``."""
    worst = 0.0
    for point in points:
        best = math.inf
        for a, b in zip(vertices[:-1], vertices[1:]):
            direction = b - a
            along = float(np.clip(np.dot(point - a, direction) / np.dot(direction, direction), 0.0, 1.0))
            best = min(best, float(np.linalg.norm(point - (a + along * direction))))
        worst = max(worst, best)
    return worst


def tool_speeds(space: JointSpace, trajectory, mask: np.ndarray) -> np.ndarray:
    """Tool speeds (m/s) between consecutive samples where ``mask`` holds, by forward kinematics (not the commanded channel)."""
    points, t = space.tool_points(trajectory), trajectory.t
    rows = np.flatnonzero(mask)
    rows = rows[rows < len(t) - 1]
    return np.linalg.norm(points[rows + 1] - points[rows], axis=1) / (t[rows + 1] - t[rows])


# ---- hidden-truth recovery for the calibration pose set: a robot whose joints are offset and whose tool is off the flange


def true_point(space: JointSpace, q: np.ndarray, offsets: np.ndarray, tool: np.ndarray) -> np.ndarray:
    pose = space.forward(np.asarray(q) + offsets)
    return pose[:3, 3] + pose[:3, :3] @ tool


def fit_offsets(space: JointSpace, stations, truth_offsets, truth_tool, noise_m: float, rng, iterations: int = 8):
    """Levenberg-Marquardt fit of the joint zero offsets and a tool translation to noisy tracker points at ``stations``
    (columns scaled by the size of error to expect, 0.01 rad and 5 mm, so the damping treats rad and m alike)."""
    n = space.dof
    scale = np.array([0.01] * n + [0.005] * 3)
    data = [true_point(space, q, truth_offsets, truth_tool) + rng.normal(0.0, noise_m, 3) for q in stations]
    params = np.zeros(n + 3)

    def model(q, p):
        return true_point(space, q, p[:n], p[n:])

    for _ in range(iterations):
        residual = np.concatenate([d - model(q, params) for d, q in zip(data, stations)])
        jacobian = np.zeros((len(residual), n + 3))
        for column in range(n + 3):
            step = np.zeros(n + 3)
            step[column] = 1e-6 * scale[column] / 0.01
            jacobian[:, column] = np.concatenate([(model(q, params + step) - model(q, params - step)) / (2 * step[column]) for q in stations]) * scale[column]
        damping = 1e-3 * float(np.trace(jacobian.T @ jacobian)) / (n + 3)
        params = params + scale * np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(n + 3), jacobian.T @ residual)
    return params


def validation_error(space: JointSpace, params, truth_offsets, truth_tool, validation) -> float:
    """RMS (m) between the fitted model's tool point and the true one over held-out configurations."""
    n = space.dof
    errors = [np.linalg.norm(true_point(space, q, params[:n], params[n:]) - true_point(space, q, truth_offsets, truth_tool)) for q in validation]
    return float(np.sqrt(np.mean(np.square(errors))))
