from __future__ import annotations

import math

import numpy as np

from ..cartesian import solve_points
from ..registry import LIMITS, kind
from ..segments import TOOL_TARGET, Motion, Piece

# Ported from robot_controllers src/robot_controllers/bench/rungs.py belt_segments (:336): from 40 mm above the surface, speed up over
# the first 20 mm to the touch speed (a quintic blend: 2 d / v), cruise the last 20 mm at it, slow to rest 10 mm below the surface (a
# quintic over 10 mm: 2 d / v), dwell 0.8 s, lift 40 mm minimum-jerk in 0.4 s. The rig's belt, its speed and the sphere tool are the
# rig's; what is kept is the one-line speed profile of the stretch.
LIFT_S = 0.4


def _samples(duration_s: float, rate_hz: float) -> np.ndarray:
    return np.arange(max(2, int(math.ceil(duration_s * rate_hz - 1e-9))) + 1) / rate_hz


def _quintic(p0: float, v0: float, p1: float, v1: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position, speed and acceleration of the quintic from (p0, v0, a = 0) to (p1, v1, a = 0) over t[-1] seconds."""
    total = float(t[-1])
    c2 = 0.0
    matrix = np.array([[total**3, total**4, total**5], [3 * total**2, 4 * total**3, 5 * total**4], [6 * total, 12 * total**2, 20 * total**3]])
    c3, c4, c5 = np.linalg.solve(matrix, [p1 - p0 - v0 * total - c2 * total**2, v1 - v0 - 2 * c2 * total, -2 * c2])
    h = p0 + v0 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    dh = v0 + 2 * c2 * t + 3 * c3 * t**2 + 4 * c4 * t**3 + 5 * c5 * t**4
    ddh = 2 * c2 + 6 * c3 * t + 12 * c4 * t**2 + 20 * c5 * t**3
    return h, dh, ddh


def _along(grid_q: np.ndarray, step: float, h: np.ndarray, dh: np.ndarray, ddh: np.ndarray):
    """Joint motion q(h(t)) from a uniform grid of joint solutions along the line (cubic Hermite in h; chain rule for qd, qdd)."""
    slope = np.gradient(grid_q, step, axis=0, edge_order=2)
    curvature = np.gradient(slope, step, axis=0, edge_order=2)
    index = np.clip(np.floor(h / step).astype(int), 0, len(grid_q) - 2)
    x = ((h - index * step) / step)[:, None]
    q0, q1, m0, m1 = grid_q[index], grid_q[index + 1], slope[index] * step, slope[index + 1] * step
    h00, h10, h01, h11 = 2 * x**3 - 3 * x**2 + 1, x**3 - 2 * x**2 + x, -2 * x**3 + 3 * x**2, x**3 - x**2
    d00, d10, d01, d11 = 6 * x**2 - 6 * x, 3 * x**2 - 4 * x + 1, -6 * x**2 + 6 * x, 3 * x**2 - 2 * x
    q = h00 * q0 + h10 * m0 + h01 * q1 + h11 * m1
    local_slope = (d00 * q0 + d10 * m0 + d01 * q1 + d11 * m1) / step
    local_curvature = curvature[index] * (1.0 - x) + curvature[index + 1] * x
    return q, local_slope * dh[:, None], local_curvature * (dh**2)[:, None] + local_slope * ddh[:, None]


@kind(use_cases=("contact_eval", "pid_tuning", "tracking_eval", "conveyor_pick_and_place"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model", "reach_map"), catalog_id="touch_down_contact")
def touch_down_contact(space, *, approach_height_m=0.04, touch_speed_m_s=0.10, touch_stretch_m=0.02, aim_below_m=0.01, dwell_s=0.8,
                       lift_m=0.04, lift_s=LIFT_S, up=(0.0, 0.0, 1.0), eef_step_m=0.001, tolerance_m=1e-4, label="touch down",
                       start=None):
    """Descend a straight tool line from ``approach_height_m`` above a surface, the last ``touch_stretch_m`` at a constant
    ``touch_speed_m_s``, aiming ``aim_below_m`` below the surface, dwell ``dwell_s``, lift ``lift_m`` (measured from the lowest
    point) in ``lift_s``: peak force, bounces and press-force error at a chosen touch speed.

    The tool starts at the approach point and the surface is ``approach_height_m`` below it along ``up`` (a direction in the base
    frame; a planar robot's plane up). The profile is the rig's rung: a quintic speed-up over the first
    ``approach_height_m - touch_stretch_m``, a cruise at the touch speed over the stretch, a quintic slow-down to rest at the aim
    point, the dwell, a minimum-jerk lift. Inverse kinematics every ``eef_step_m`` along the line, refused at the first sample
    that fails. WHY the touch speed is never stretched: it is the stated number; a profile the joint limits cannot follow is
    refused naming the joint, not slowed. The tool keeps its start orientation. Task-agnostic: a height, a speed, a depth."""
    direction = np.asarray(up, float)
    if np.linalg.norm(direction) < 1e-12:
        raise ValueError("up must be a nonzero direction")
    direction = direction / np.linalg.norm(direction)
    if not (0.0 < touch_stretch_m < approach_height_m) or touch_speed_m_s <= 0 or aim_below_m <= 0 or dwell_s < 0:
        raise ValueError("need 0 < touch_stretch_m < approach_height_m, touch_speed_m_s > 0, aim_below_m > 0 and dwell_s >= 0")
    bottom = approach_height_m + aim_below_m
    if not 0.0 < lift_m <= bottom:
        raise ValueError(f"lift_m must be in (0, {bottom:.3f}] m (the lift stays below where the tool started)")
    rate = space.rate_hz
    origin_pose = space.forward(start)
    count = max(51, int(math.ceil(bottom / eef_step_m)) + 1)
    step = bottom / (count - 1)
    depths = np.arange(count) * step
    poses = []
    for depth in depths[1:]:
        pose = origin_pose.copy()
        pose[:3, 3] = origin_pose[:3, 3] - depth * direction
        poses.append(pose)
    solved = solve_points(space, start, poses, what=label, tolerance_m=tolerance_m)  # the first failing sample is named
    grid = np.vstack([start, *solved])
    jump = np.max(np.abs(np.diff(grid, axis=0)))
    if jump > 0.1:
        raise ValueError(f"{label}: a joint jumps {math.degrees(jump):.1f}° between samples {eef_step_m * 1000:.1f} mm apart: a branch flip or a singularity")

    def point(h: np.ndarray) -> np.ndarray:
        return origin_pose[:3, 3] - h[:, None] * direction

    phases = []  # (label, h, dh, ddh)
    cruise_start = approach_height_m - touch_stretch_m
    t = _samples(2.0 * cruise_start / touch_speed_m_s, rate)
    phases.append(("speed up", *_quintic(0.0, 0.0, cruise_start, touch_speed_m_s, t)))
    t = _samples(touch_stretch_m / touch_speed_m_s, rate)
    h_cruise = cruise_start + touch_speed_m_s * t
    phases.append(("touch stretch", h_cruise, np.full_like(t, touch_speed_m_s), np.zeros_like(t)))
    # WHY from where the cruise really ended: its last sample is on the rate's grid, up to one sample past the stretch
    t = _samples(2.0 * (bottom - h_cruise[-1]) / touch_speed_m_s, rate)
    phases.append(("press below the surface", *_quintic(float(h_cruise[-1]), touch_speed_m_s, bottom, 0.0, t)))
    if dwell_s > 0:
        t = _samples(dwell_s, rate)
        phases.append(("dwell in contact", np.full_like(t, bottom), np.zeros_like(t), np.zeros_like(t)))
    t = _samples(lift_s, rate)
    phases.append(("lift", *_quintic(bottom, 0.0, bottom - lift_m, 0.0, t)))
    pieces, limits = [], space.limits
    for name, h, dh, ddh in phases:
        t = np.arange(len(h)) / rate
        q, qd, qdd = _along(grid, step, np.clip(h, 0.0, bottom), dh, ddh)
        for quantity, values, limit in (("velocity", qd, limits.max_velocity), ("acceleration", qdd, limits.max_acceleration)):
            ratio = np.abs(values) / limit
            if np.max(ratio) > 1.0 + 1e-6:
                joint = int(np.argmax(np.max(ratio, axis=0)))
                raise ValueError(f"{label}: {name} needs {space.joint_names[joint]} {quantity} at {np.max(ratio):.2f}x its limit; "
                                 f"the touch speed is the stated number, so lower touch_speed_m_s (or the stretch) instead")
        pieces.append(Piece(t, q, qd, qdd, f"{label}: {name}", channels={TOOL_TARGET: point(h)}))
    surface_h = approach_height_m
    return Motion(pieces, (), {"surface_depth_m": float(surface_h), "aim_depth_m": float(bottom), "touch_speed_m_s": float(touch_speed_m_s)})
