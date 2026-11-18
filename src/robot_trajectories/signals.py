"""Signals this package adds beside the shared kernels (the round's math, vendored byte for byte, so they stay as they are).

``constant_speed_chirp``: the catalog's amplitude rule for a chirp, ``angle amplitude = v / (2 pi f(t))``.

WHY a new function and not ``kernels.chirp``: the shared chirp builds the sweep at amplitude 1 and scales the WHOLE sweep
by one gain, so its highest frequency decides the amplitude everywhere. On the pendulum (7.39 rad/s^2) the 5 Hz end allows
about 0.4 deg, and the shared chirp then moves 0.34 deg at 0.2 Hz too, where 16 deg would fit. The rack runner's chirp
(experiments.py ``chirp``), PID tuna's identification chirp and Beckhoff's sweep all hold the SPEED amplitude instead,
so the angle shrinks as the frequency rises. Here that rule is capped, at every instant, by the room and the
acceleration limit (the rack's ``chirp_log`` takes the smallest of the three the same way), tapered in and out, and
finally checked: a last uniform gain keeps every sample inside the position, velocity and acceleration limits.
"""

from __future__ import annotations

import math

import numpy as np

from . import kernels as k

RULES = ("constant_speed", "constant_angle")
# WHY a smooth minimum (p = 8) and not min(): the amplitude switches from the room cap to the speed or acceleration cap
# mid-sweep; a hard min() has a corner there, and a corner in the amplitude is a jump in velocity (d/dt of A sin phi).
SOFT_MIN_POWER = 8.0


def _samples(duration_s: float, rate_hz: float) -> np.ndarray:
    """The kernels' time grid (a whole number of samples), so this piece joins theirs on one clock."""
    steps = max(2, int(math.ceil(duration_s * rate_hz - 1e-9)))
    return np.arange(steps + 1) / rate_hz


def _taper(t: np.ndarray, total: float, taper_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raised-cosine in and out, with its first two derivatives.

    WHY raised cosine and not the runner's linear 0.5 s taper: a linear taper's corner is a velocity step, which a
    trajectory checked against its acceleration limit cannot contain."""
    w, dw, ddw = np.ones_like(t), np.zeros_like(t), np.zeros_like(t)
    taper_s = min(taper_s, 0.5 * total)
    if taper_s <= 0:
        return w, dw, ddw
    for mask, local, sign in ((t < taper_s, t, 1.0), (t > total - taper_s, total - t, -1.0)):
        x = local[mask] / taper_s
        w[mask] = 0.5 - 0.5 * np.cos(np.pi * x)
        dw[mask] = sign * 0.5 * np.pi / taper_s * np.sin(np.pi * x)
        ddw[mask] = 0.5 * (np.pi / taper_s) ** 2 * np.cos(np.pi * x)
    return w, dw, ddw


def _taper_needed(taper_s: float, total: float, envelope: np.ndarray, omega: np.ndarray, omega_dot: np.ndarray,
                  acceleration_limit: float) -> float:
    """The taper, lengthened (up to a tenth of the sweep) until fading the swing in and out fits the acceleration limit.

    WHY: a raised cosine over tau adds up to A (pi^2 / 2 tau^2 + pi w / tau) of acceleration on top of A w^2. At the
    pendulum's 0.5 Hz start (13.6 deg, room-bound) the catalog's 0.5 s taper needs 1.2x the limit, and the last-resort
    uniform gain would then shrink the whole sweep by 15 %; a 0.9 s taper costs nothing. Solved from the bound
    A (pi^2 / 2 x^2 ... ) <= a with x = 1 / tau, at both ends."""
    longest = max(taper_s, 0.1 * total)
    needed = taper_s
    for index in (0, -1):
        amplitude, rate = float(envelope[index]), float(omega[index])
        headroom = acceleration_limit / max(amplitude, 1e-12) - rate**2 - abs(float(omega_dot[index]))
        if headroom <= 0:
            return longest
        needed = max(needed, math.pi / (math.sqrt(rate**2 + 2 * headroom) - rate))
    return min(needed, longest, 0.5 * total)


def constant_speed_chirp(centre, joint: int, bounds: k.Bounds, rate_hz: float = k.DEFAULT_RATE_HZ, *, start_hz: float = 0.5,
                         end_hz: float = 2.0, duration_s: float = 20.0, logarithmic: bool = False, speed_fraction: float = 0.8,
                         room_fraction: float = 0.85, acceleration_fraction: float = 0.85, taper_s: float = 0.5,
                         rule: str = "constant_speed", label: str = "linear chirp") -> tuple[k.Arrays, dict]:
    """A swept sine on one joint whose angle amplitude follows ``rule``, inside every limit; returns (arrays, info).

    - ``constant_speed``: A(t) = the smooth minimum of v / w(t), the room and a / w(t)^2, where v = ``speed_fraction`` x
      the velocity limit, the room = ``room_fraction`` x the distance to the nearer position limit, a =
      ``acceleration_fraction`` x the acceleration limit, w = 2 pi f(t).
    - ``constant_angle``: one amplitude for the whole sweep, the smallest of those caps at the top frequency (BAM's
      sin(t^2) shape; what ``kernels.chirp`` does).

    ``info`` holds the amplitude envelope, the taper used (``taper_s``, lengthened up to a tenth of the sweep when the
    fade itself would break the acceleration limit) and the final safety gain (1.0 unless that was not enough).
    """
    if rule not in RULES:
        raise ValueError(f"amplitude rule {rule!r}: choose one of {RULES}")
    if start_hz <= 0 or end_hz <= 0:
        raise ValueError(f"chirp frequencies must be positive, got {start_hz} and {end_hz} Hz")
    centre = np.asarray(centre, float)
    t = _samples(duration_s, rate_hz)
    total = t[-1]
    if logarithmic:
        growth = math.log(end_hz / start_hz) / total
        omega = 2 * np.pi * start_hz * np.exp(growth * t)
        phase = 2 * np.pi * start_hz * (np.exp(growth * t) - 1) / growth if growth else 2 * np.pi * start_hz * t
        omega_dot = growth * omega
    else:
        omega = 2 * np.pi * (start_hz + (end_hz - start_hz) * t / total)
        phase = 2 * np.pi * (start_hz * t + 0.5 * (end_hz - start_hz) * t**2 / total)
        omega_dot = np.full_like(t, 2 * np.pi * (end_hz - start_hz) / total)

    room_full = float(min(centre[joint] - bounds.low[joint], bounds.high[joint] - centre[joint]))
    velocity_limit, acceleration_limit = float(bounds.max_velocity[joint]), float(bounds.max_acceleration[joint])
    room = room_fraction * room_full
    speed = speed_fraction * velocity_limit
    acceleration = acceleration_fraction * acceleration_limit
    if rule == "constant_speed":
        caps = np.vstack([speed / omega, np.full_like(t, room), acceleration / omega**2])
        envelope = np.sum(caps ** -SOFT_MIN_POWER, axis=0) ** (-1.0 / SOFT_MIN_POWER)
        envelope_dot = np.gradient(envelope, t, edge_order=2)
        envelope_ddot = np.gradient(envelope_dot, t, edge_order=2)
    else:
        top = float(np.max(omega))
        envelope = np.full_like(t, min(room, speed / top, acceleration / top**2))
        envelope_dot, envelope_ddot = np.zeros_like(t), np.zeros_like(t)
    taper_s = _taper_needed(taper_s, total, envelope, omega, omega_dot, acceleration_limit)
    w, dw, ddw = _taper(t, total, taper_s)
    amplitude = envelope * w
    amplitude_dot = envelope_dot * w + envelope * dw
    amplitude_ddot = envelope_ddot * w + 2 * envelope_dot * dw + envelope * ddw
    sine, cosine = np.sin(phase), np.cos(phase)
    x = amplitude * sine
    dx = amplitude_dot * sine + amplitude * omega * cosine
    ddx = (amplitude_ddot * sine + 2 * amplitude_dot * omega * cosine + amplitude * omega_dot * cosine
           - amplitude * omega**2 * sine)
    # WHY a last check against the FULL limits: the taper and the sweep add slope terms the caps did not see; a uniform
    # gain below 1 trims the whole signal rather than clipping a sample (which would be a jump).
    worst = max(float(np.max(np.abs(x))) / room_full, float(np.max(np.abs(dx))) / velocity_limit,
                float(np.max(np.abs(ddx))) / acceleration_limit)
    gain = min(1.0, (1.0 - 1e-9) / worst) if worst > 0 else 1.0
    q = np.tile(centre, (len(t), 1))
    qd, qdd = np.zeros_like(q), np.zeros_like(q)
    q[:, joint] += gain * x
    qd[:, joint] = gain * dx
    qdd[:, joint] = gain * ddx
    info = {"rule": rule, "gain": gain, "taper_s": taper_s, "frequency_hz": omega / (2 * np.pi), "envelope_rad": gain * envelope}
    return k.Arrays(t, q, qd, qdd, label), info

