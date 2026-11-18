from __future__ import annotations

import math

import numpy as np

from .. import kernels as k
from ..registry import kind
from ..segments import Motion
from ._lane_a import deg, joint_room, single_joint

# The rack's dither lines (Hz) and where they sit in its 1-6 Hz band: schedule.py DITHER_LINES_HZ (1.0, 1.7, 2.9, 4.1, 6.0).
LINE_POSITIONS = (0.0, 0.14, 0.38, 0.62, 1.0)


def _cosine_ramp(duration: float, v0: float, v1: float):
    """(velocity, position gained) at time tau of a raised-cosine change of speed from v0 to v1 over ``duration``."""
    def at(tau):
        s = np.clip(tau / duration, 0.0, 1.0)
        velocity = v0 + (v1 - v0) * (1 - np.cos(np.pi * s)) / 2
        position = (v0 + v1) / 2 * np.clip(tau, 0.0, duration) - (v1 - v0) * duration / (2 * np.pi) * np.sin(np.pi * s)
        acceleration = np.where((tau >= 0) & (tau <= duration), (v1 - v0) * np.pi / (2 * duration) * np.sin(np.pi * s), 0.0)
        return velocity, position, acceleration
    return at


@kind(use_cases=("inertia_id", "friction_id"), catalog_id="biased_dither")
def biased_dither(space, joint=None, *, ramp_speed_fraction=0.075, lines_low_hz=1.0, lines_high_hz=6.0,
                  ripple_speed_fraction_of_ramp=0.8, range_fraction=0.85, turn_s=0.6, ripple_fade_s=0.5, first_direction=1, seed=0,
                  label="biased dither", start=None):
    """A slow constant-speed ramp 0 -> +A -> -A -> 0 carrying a small multisine (five lines across ``lines_low_hz`` ..
    ``lines_high_hz``, random phases from ``seed``) whose speed never exceeds ``ripple_speed_fraction_of_ramp`` of the ramp's:
    the joint slides without reversing, so inertia and viscous friction are read with no stick-slip (Wernholt's offset, the
    rack's 'dither'; ``first_direction`` -1 goes to the - side first, the rack's second variant).

    ``ramp_speed_fraction``: the ramp's speed as a fraction of the velocity limit. ``range_fraction``: A as a fraction of the room
    on the smaller side. ``turn_s``: a raised-cosine turn at each end of the slide (the only reversals).

    WHY the ramp is constant-speed with raised-cosine turns and the ripple is windowed to its constant-speed stretches (the rack
    built the ramp from minimum-jerk legs between knots, whose speed falls to zero at the knots and peaks at 1.875x the ramp
    speed mid-leg): then the ripple's speed budget is exact at every sample the ripple is on, including the window's own slope, and
    the only sign changes of the joint's speed are the two turns, which a test counts."""
    if first_direction not in (1, -1):
        raise ValueError(f"biased_dither.first_direction is +1 or -1, got {first_direction!r}")
    if not (0 < ramp_speed_fraction <= 1 and 0 < ripple_speed_fraction_of_ramp < 1 and 0 < range_fraction <= 1
            and 0 < lines_low_hz <= lines_high_hz and turn_s > 0 and ripple_fade_s > 0):
        raise ValueError("biased_dither needs 0 < ramp_speed_fraction <= 1, 0 < ripple_speed_fraction_of_ramp < 1, 0 < range_fraction <= 1, "
                         "0 < lines_low_hz <= lines_high_hz, turn_s > 0 and ripple_fade_s > 0")
    index = space.joint_index(joint)
    bounds = space.room(start)
    amplitude = range_fraction * joint_room(start, bounds, index)
    v = ramp_speed_fraction * float(bounds.max_velocity[index])
    sign = float(first_direction)
    pull_back = v * turn_s / math.pi  # how far the turn carries the joint past where the reversal starts, per half
    # the ramp, as (duration, speed in, speed out) pieces: speed up, slide, turn, slide, turn, slide, slow down
    ahead = amplitude - pull_back - v * turn_s / 2
    last = amplitude - pull_back - v * turn_s / 2
    if ahead <= 0 or last <= 0:
        raise ValueError(f"biased_dither: the range ({deg(amplitude)}) is too small for turn_s {turn_s:g} s at {v:.3g} rad/s: lower turn_s or the ramp speed")
    plan = [(turn_s, 0.0, v), (ahead / v, v, v), (turn_s, v, -v), ((2 * amplitude - 2 * pull_back) / v, -v, -v), (turn_s, -v, v),
            (last / v, v, v), (turn_s, v, 0.0)]
    edges = np.concatenate([[0.0], np.cumsum([piece[0] for piece in plan])])
    total = float(edges[-1])
    t = k._samples(total + 1.0 / space.rate_hz, space.rate_hz)
    x, dx, ddx = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    position = 0.0
    for number, (duration, v0, v1) in enumerate(plan):
        inside = (t >= edges[number]) & (t < edges[number + 1]) if number < len(plan) - 1 else (t >= edges[number])
        tau = t[inside] - edges[number]
        if v0 == v1:
            x[inside], dx[inside] = position + v0 * tau, v0
            gained = v0 * duration
        else:
            velocity, gain_x, acceleration = _cosine_ramp(duration, v0, v1)(tau)
            x[inside], dx[inside], ddx[inside] = position + gain_x, velocity, acceleration
            gained = (v0 + v1) / 2 * duration
        position += gained
    # after the last piece the joint rests at 0 (the plan returns there exactly)
    x[t >= total], dx[t >= total], ddx[t >= total] = 0.0, 0.0, 0.0
    # the ripple: only on the constant-speed stretches (plan pieces 1, 3 and 5), raised-cosine windowed
    slides = [(edges[number], edges[number + 1]) for number in (1, 3, 5)]
    window, d_window, dd_window = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    skipped = []
    for number, (a, b) in enumerate(slides):
        # WHY a fade of at most a quarter of the slide: on a small range the slides are short, and a window longer than the slide
        # would leave no ripple at all; a shorter fade steepens the window, which the ripple's speed budget below accounts for.
        fade = min(ripple_fade_s, 0.25 * (b - a))
        if fade < 0.1:
            skipped.append(number + 1)
            continue
        for lo, hi, rising in ((a, a + fade, True), (b - fade, b, False)):
            mask = (t >= lo) & (t <= hi)
            s = (t[mask] - lo) / fade
            direction = 1.0 if rising else -1.0
            window[mask] = 0.5 - 0.5 * direction * np.cos(np.pi * s)
            d_window[mask] = 0.5 * direction * np.pi / fade * np.sin(np.pi * s)
            dd_window[mask] = 0.5 * (np.pi / fade) ** 2 * np.cos(np.pi * s) * direction
        window[(t > a + fade) & (t < b - fade)] = 1.0
    lines = lines_low_hz + (lines_high_hz - lines_low_hz) * np.array(LINE_POSITIONS)
    omega = 2 * np.pi * lines
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(lines))
    # each line's angle amplitude: the ripple's speed (its lines at full tilt plus the window's slope on the lines' swing) stays
    # within ripple_speed_fraction_of_ramp of the ramp's; and its acceleration within 0.8 of what the ramp's turns leave
    window_slope = float(np.max(np.abs(d_window))) if np.any(window) else 0.0
    each = ripple_speed_fraction_of_ramp * v / (float(np.sum(omega)) + len(lines) * window_slope)
    ripple_acceleration = float(np.sum(omega**2)) + 2 * len(lines) * window_slope * float(np.max(omega)) \
        + len(lines) * float(np.max(np.abs(dd_window)))
    room_for_acceleration = 0.8 * float(bounds.max_acceleration[index]) - math.pi * v / turn_s
    note = ""
    if room_for_acceleration <= 0:
        raise ValueError("biased_dither: the ramp's own turns already use the acceleration limit: lengthen turn_s")
    if each * ripple_acceleration > room_for_acceleration:
        scale = room_for_acceleration / (each * ripple_acceleration)
        each *= scale
        note = f" (ripple lowered x{scale:.3f} to leave the acceleration limit its margin: the ramp is untouched)"
    dither = sum(each * np.sin(w * t + p) for w, p in zip(omega, phases, strict=True))
    d_dither = sum(each * w * np.cos(w * t + p) for w, p in zip(omega, phases, strict=True))
    dd_dither = sum(-each * w**2 * np.sin(w * t + p) for w, p in zip(omega, phases, strict=True))
    x_total = sign * x + window * dither
    dx_total = sign * dx + d_window * dither + window * d_dither
    ddx_total = sign * ddx + dd_window * dither + 2 * d_window * d_dither + window * dd_dither
    if skipped:
        note += f" (no ripple on slide {', '.join(map(str, skipped))}: shorter than 0.4 s)"
    summary = (f"{label}: ramp {math.degrees(v):.2f}°/s over +-{deg(amplitude)}, five lines {lines_low_hz:g}-{lines_high_hz:g} Hz at "
               f"{deg(each)} each (ripple speed <= {ripple_speed_fraction_of_ramp:g} of the ramp's){note}")
    return Motion([single_joint(t, start, index, x_total, dx_total, ddx_total, label)], [summary],
                  entry={"slides_s": [[float(a), float(b)] for a, b in slides], "ramp_speed_rad_s": v})
