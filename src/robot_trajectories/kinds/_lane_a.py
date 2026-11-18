"""Lane A's shared math (excitation and controller-test kinds): imported by its kinds, registers nothing.

Everything here is numpy-only and returns ``kernels.Arrays`` pieces on the kernels' own sample grid, so a lane-A kind joins the
existing ones on one clock.
"""

from __future__ import annotations

import math

import numpy as np

from .. import kernels as k


def single_joint(t: np.ndarray, centre, joint: int, x: np.ndarray, dx: np.ndarray, ddx: np.ndarray, label: str) -> k.Arrays:
    """One joint moves by (x, dx, ddx) about ``centre``; every other joint is held there."""
    q = np.tile(np.asarray(centre, float), (len(t), 1))
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    q[:, joint] += x
    qd[:, joint] = dx
    qdd[:, joint] = ddx
    return k.Arrays(t, q, qd, qdd, label)


def joint_room(centre, bounds: k.Bounds, joint: int) -> float:
    """The distance from ``centre`` to the nearer position limit of one joint (the room both ways)."""
    return float(min(centre[joint] - bounds.low[joint], bounds.high[joint] - centre[joint]))


def limit_gain(room: float, bounds: k.Bounds, joint: int, x: np.ndarray, dx: np.ndarray, ddx: np.ndarray, margin: float) -> float:
    """The gain that puts the tightest of room / velocity / acceleration of (x, dx, ddx) at ``margin`` of its limit
    (kernels._scaled's rule for one joint, returned instead of applied so a caller can cap it)."""
    ratios = (float(np.max(np.abs(x))) / max(room, 1e-12), float(np.max(np.abs(dx))) / float(bounds.max_velocity[joint]),
              float(np.max(np.abs(ddx))) / float(bounds.max_acceleration[joint]))
    return margin / max(max(ratios), 1e-12)


def deg(value: float) -> str:
    return f"{math.degrees(value):.3g}°"


# ====================================================================== level sequences (PRBS, doublet, 3-2-1-1)


def level_sequence(centre, joint: int, bounds: k.Bounds, offsets_rad, durations_s, rate_hz: float, *, edge_fraction: float,
                   margin: float, label: str) -> tuple[k.Arrays, float]:
    """The joint steps through the levels ``centre + gain * offsets`` for ``durations_s`` each, every edge a minimum-jerk
    transition of ``edge_fraction`` of the shortest level; it starts at ``centre`` (the first edge goes there) and ends back at
    it at rest (one more edge). Returns the motion and the ``gain`` (<= 1) that made the biggest edge fit the limits.

    WHY the gain is solved and the timing is not: a level's length sets the signal's spectrum (PRBS: flat to 0.44 / clock
    period); a position loop can only follow edges the limits allow, so the AMPLITUDE gives way and the caller says so."""
    if not 0.0 < edge_fraction <= 1.0:
        raise ValueError(f"edge_fraction must lie in (0, 1], got {edge_fraction}")
    offsets = np.asarray(offsets_rad, float)
    counts = np.array([max(2, int(round(duration * rate_hz))) for duration in durations_s])
    edge = max(1, int(round(edge_fraction * int(counts.min()))))
    edge_s = edge / rate_hz
    levels = np.concatenate([[0.0], offsets, [0.0]])
    biggest = float(np.max(np.abs(np.diff(levels))))
    room = joint_room(centre, bounds, joint)
    gain = min(1.0, margin * room / max(float(np.max(np.abs(offsets))), 1e-12),
               edge_s * margin * float(bounds.max_velocity[joint]) / (k.MIN_JERK_PEAK_VELOCITY * biggest),
               edge_s**2 * margin * float(bounds.max_acceleration[joint]) / (k.MIN_JERK_PEAK_ACCELERATION * biggest))
    levels = gain * levels
    counts_all = np.concatenate([counts, [edge]])  # the last slot is the edge back to the centre
    starts = np.concatenate([[0], np.cumsum(counts_all)])
    samples = np.arange(int(starts[-1]) + 1)
    slot = np.minimum(np.searchsorted(starts, samples, side="right") - 1, len(counts_all) - 1)
    s = np.clip((samples - starts[slot]) / edge, 0.0, 1.0)  # at s = 1 the edge's velocity and acceleration are exactly 0
    delta = (levels[slot + 1] - levels[slot])
    x = levels[slot] + delta * (10 * s**3 - 15 * s**4 + 6 * s**5)
    dx = delta * (30 * s**2 - 60 * s**3 + 30 * s**4) / edge_s
    ddx = delta * (60 * s - 180 * s**2 + 120 * s**3) / edge_s**2
    return single_joint(samples / rate_hz, centre, joint, x, dx, ddx, label), gain


# ====================================================================== quintic between states (rapid replanning)


def quintic(p0: np.ndarray, v0: np.ndarray, a0: np.ndarray, p1: np.ndarray, v1: np.ndarray, a1: np.ndarray, total_s: float,
            t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The quintic from (p0, v0, a0) at t = 0 to (p1, v1, a1) at ``total_s``, evaluated at ``t``: three (len(t), J) arrays."""
    T = total_s
    delta = p1 - p0 - v0 * T - 0.5 * a0 * T**2
    dv = v1 - v0 - a0 * T
    da = a1 - a0
    c3 = (10 * delta - 4 * dv * T + 0.5 * da * T**2) / T**3
    c4 = (-15 * delta + 7 * dv * T - da * T**2) / T**4
    c5 = (6 * delta - 3 * dv * T + 0.5 * da * T**2) / T**5
    t = np.asarray(t, float)[:, None]
    p = p0 + v0 * t + 0.5 * a0 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    v = v0 + a0 * t + 3 * c3 * t**2 + 4 * c4 * t**3 + 5 * c5 * t**4
    a = a0 + 6 * c3 * t + 12 * c4 * t**2 + 20 * c5 * t**3
    return p, v, a


# ====================================================================== maximal-length sequences (PRBS)

# WHY these taps: the feedback stages of a maximal-length linear-feedback shift register (period 2^n - 1), from the standard table
# (Davies 1970; MATLAB idinput's prbs): x^n + x^(n-1) + 1 style primitive polynomials. tests/test_lane_a.py checks the period and the
# balance (2^(n-1) ones) of every entry, so a wrong tap cannot pass.
MAXIMAL_TAPS: dict[int, tuple[int, ...]] = {
    3: (3, 2), 4: (4, 3), 5: (5, 3), 6: (6, 5), 7: (7, 6), 8: (8, 6, 5, 4), 9: (9, 5), 10: (10, 7), 11: (11, 9), 12: (12, 11, 10, 4),
    13: (13, 12, 11, 8), 14: (14, 13, 12, 2), 15: (15, 14), 16: (16, 15, 13, 4),
}


def m_sequence(stages: int, seed: int = 0) -> np.ndarray:
    """One period (2^stages - 1 bits) of a maximal-length sequence; ``seed`` picks the register's (non-zero) starting fill."""
    if stages not in MAXIMAL_TAPS:
        raise ValueError(f"register_stages must be one of {sorted(MAXIMAL_TAPS)}, got {stages}")
    period = 2**stages - 1
    fill = 1 + int(seed) % period
    register = [(fill >> index) & 1 for index in range(stages)]
    bits = np.zeros(period, dtype=int)
    for index in range(period):
        bits[index] = register[-1]
        feedback = 0
        for tap in MAXIMAL_TAPS[stages]:
            feedback ^= register[tap - 1]
        register = [feedback, *register[:-1]]
    return bits
