from __future__ import annotations

import math

from .. import kernels as k
from ..registry import kind
from ..segments import Motion
from ..signals import RULES, constant_speed_chirp


@kind(use_cases=("inertia_id", "bandwidth", "delay_id", "pid_tuning", "state_coverage"), catalog_id="linear_chirp")
def linear_chirp(space, joint=None, *, f_start_hz=0.5, f_end_hz=2.0, duration_s=20.0, speed_amplitude_fraction=0.8,
                 amplitude_rule="constant_speed", taper_s=0.5, hold_after_s=1.0, room_fraction=0.85, acceleration_fraction=0.85,
                 label="linear chirp", start=None):
    """A sine whose frequency rises linearly from f_start to f_end, its amplitude held at a constant speed (angle = v / 2 pi f,
    capped by the room and the acceleration limit) or a constant angle, tapered in and out, then a hold (the rack runner's
    chirp; PID tuna's identification chirp).

    ``speed_amplitude_fraction``: v as a fraction of the velocity limit. ``room_fraction`` / ``acceleration_fraction``: the
    caps (the rack's 0.85). ``taper_s``: the raised-cosine taper, lengthened when the fade itself needs it.

    WHY the defaults stop at 2 Hz (the catalog row's note): PID tuna's chirp stops at command_rate / 50; between commands
    moteus integrates a constant velocity, so a faster chirp breaks J qdd = tau (measured 2026-10-25: 5.7 Hz read a JMC's
    inertia +50 %, 2 Hz -3 %)."""
    if amplitude_rule not in RULES:
        raise ValueError(f"linear_chirp.amplitude_rule must be one of {RULES}, got {amplitude_rule!r}")
    if f_start_hz <= 0 or f_end_hz <= 0:
        raise ValueError(f"linear_chirp frequencies must be positive, got {f_start_hz} and {f_end_hz} Hz")
    sweep, info = constant_speed_chirp(start, space.joint_index(joint), space.room(start), space.rate_hz, start_hz=f_start_hz,
                                       end_hz=f_end_hz, duration_s=duration_s, speed_fraction=speed_amplitude_fraction,
                                       room_fraction=room_fraction, acceleration_fraction=acceleration_fraction, taper_s=taper_s,
                                       rule=amplitude_rule, label=label)
    envelope = info["envelope_rad"]
    note = (f"{label}: swing {math.degrees(float(envelope[0])):.1f}° at {f_start_hz:g} Hz -> "
            f"{math.degrees(float(envelope[-1])):.2f}° at {f_end_hz:g} Hz ({amplitude_rule.replace('_', ' ')})"
            + (f", taper {info['taper_s']:.2f} s (not {taper_s:g}) so the fade fits the acceleration limit"
               if info["taper_s"] > taper_s + 1e-9 else "")
            + (f", trimmed x{info['gain']:.3f} to stay inside the limits" if info["gain"] < 0.999 else ""))
    pieces = [sweep]
    if hold_after_s > 0:
        pieces.append(k.hold(start, hold_after_s, space.rate_hz, label=f"{label}: hold after"))
    return Motion(pieces, [note])
