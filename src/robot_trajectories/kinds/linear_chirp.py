from __future__ import annotations

import math

from .. import kernels as k

from ..kind import Context, Kind, param
from ..signals import RULES, constant_speed_chirp


class LinearChirp(Kind):
    """A sine whose frequency rises linearly from f_start to f_end, its amplitude held at a constant speed
    (angle = v / 2 pi f, capped by the room and the acceleration limit) or a constant angle, tapered in and out, then a
    hold (the rack runner's chirp; PID tuna's identification chirp).

    WHY the defaults stop at 2 Hz (the catalog row's note): PID tuna's chirp stops at command_rate / 50; between commands
    moteus integrates a constant velocity, so a faster chirp breaks J qdd = tau (measured 2026-10-25: 5.7 Hz read a JMC's
    inertia +50 %, 2 Hz -3 %)."""

    catalog_id = "linear_chirp"
    family = "excitation"
    use_cases = ("inertia_id", "bandwidth", "delay_id", "pid_tuning", "state_coverage")

    joint: str | int | None = param(None, "joint", "the joint that moves (name or index; optional on a 1-DOF robot)")
    f_start_hz: float = param(0.5, "Hz", "the sweep's first frequency")
    f_end_hz: float = param(2.0, "Hz", "the sweep's last frequency")
    duration_s: float = param(20.0, "s", "the sweep's length, tapers included")
    speed_amplitude_fraction: float = param(0.8, "1", "the speed amplitude v as a fraction of the velocity limit")
    amplitude_rule: str = param("constant_speed", "text", "constant_speed (angle = v / 2 pi f) or constant_angle")
    taper_s: float = param(0.5, "s", "raised-cosine taper in and out (at least; longer when the fade needs it)")
    hold_after_s: float = param(1.0, "s", "the hold after the sweep (the runner's)")
    room_fraction: float = param(0.85, "1", "cap: this fraction of the room to the nearer position limit (the rack's 0.85)")
    acceleration_fraction: float = param(0.85, "1", "cap: this fraction of the acceleration limit (the rack's chirp_log)")

    def check(self) -> None:
        if self.amplitude_rule not in RULES:
            raise ValueError(f"LinearChirp.amplitude_rule must be one of {RULES}, got {self.amplitude_rule!r}")
        if self.f_start_hz <= 0 or self.f_end_hz <= 0:
            raise ValueError(f"LinearChirp frequencies must be positive, got {self.f_start_hz} and {self.f_end_hz} Hz")

    def pieces(self, context: Context) -> list[k.Arrays]:
        joint = context.joint(self.joint)
        sweep, info = constant_speed_chirp(context.start, joint, context.room_around(context.start), context.rate_hz,
                                           start_hz=self.f_start_hz, end_hz=self.f_end_hz, duration_s=self.duration_s,
                                           speed_fraction=self.speed_amplitude_fraction, room_fraction=self.room_fraction,
                                           acceleration_fraction=self.acceleration_fraction, taper_s=self.taper_s,
                                           rule=self.amplitude_rule, label=self.display_label)
        envelope = info["envelope_rad"]
        context.note(f"{self.display_label}: swing {math.degrees(float(envelope[0])):.1f}° at {self.f_start_hz:g} Hz -> "
                     f"{math.degrees(float(envelope[-1])):.2f}° at {self.f_end_hz:g} Hz ({self.amplitude_rule.replace('_', ' ')})"
                     + (f", taper {info['taper_s']:.2f} s (not {self.taper_s:g}) so the fade fits the acceleration limit"
                        if info["taper_s"] > self.taper_s + 1e-9 else "")
                     + (f", trimmed x{info['gain']:.3f} to stay inside the limits" if info["gain"] < 0.999 else ""))
        pieces = [sweep]
        if self.hold_after_s > 0:
            pieces.append(k.hold(context.start, self.hold_after_s, context.rate_hz, label=f"{self.display_label}: hold after"))
        return pieces
