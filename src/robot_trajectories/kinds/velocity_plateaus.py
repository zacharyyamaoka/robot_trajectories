from __future__ import annotations

import math

from .. import kernels as k

from ..kind import Context, Kind, param


class VelocityPlateaus(Kind):
    """Constant-speed plateaus out and back from the start at a ladder of speeds, each held long enough to settle:
    friction against speed (Coulomb, viscous, Stribeck, breakaway). The rack schedule's ``plateaus`` family.

    Faithful to schedule.py: one speed per run, out and back, each plateau as long as the room allows (at most
    ``plateau_s``); a speed whose plateau would be shorter than ``min_plateau_s`` is skipped (the sines carry it) and the
    program notes say so. WHY ramps where the rack steps the velocity: a velocity step is not inside an acceleration
    limit; the ramps take ``acceleration_fraction`` of it (the kernels' velocity sweep uses 0.8) and their distance comes
    out of the tenth of the room the rack's rule leaves spare."""

    catalog_id = "velocity_plateaus"
    family = "quasi_static"
    use_cases = ("friction_id", "speed_limits", "state_coverage")

    joint: str | int | None = param(None, "joint", "the joint that moves (name or index; optional on a 1-DOF robot)")
    speed_fractions: tuple[float, ...] = param((0.025, 0.05, 0.1, 0.175, 0.25, 0.35), "1",
                                               "each plateau's speed as a fraction of the velocity limit")
    plateau_s: float = param(2.0, "s", "the longest plateau")
    min_plateau_s: float = param(0.8, "s", "a speed whose plateau would be shorter is skipped")
    first_direction: str = param("+", "text", "'+' or '-': the side each run goes to first")
    acceleration_fraction: float = param(0.8, "1", "the ramps' acceleration, as a fraction of the limit")
    dwell_s: float = param(0.5, "s", "the hold between runs (the rack stops and settles between them)")

    def check(self) -> None:
        if self.first_direction not in ("+", "-"):
            raise ValueError(f"VelocityPlateaus.first_direction is '+' or '-', got {self.first_direction!r}")

    def pieces(self, context: Context) -> list[k.Arrays]:
        joint = context.joint(self.joint)
        room = context.room_around(context.start)
        sign = 1.0 if self.first_direction == "+" else -1.0
        side = (room.high[joint] - context.start[joint]) if sign > 0 else (context.start[joint] - room.low[joint])
        ramp_acceleration = self.acceleration_fraction * float(context.bounds.max_acceleration[joint])
        pieces: list[k.Arrays] = []
        skipped = []
        for fraction in self.speed_fractions:
            speed = fraction * float(context.bounds.max_velocity[joint])
            # the rack's rule (plateau <= 0.9 x room / speed); the ramps then come out of the last tenth, and only if they
            # do not fit there does the plateau shorten
            plateau = min(self.plateau_s, 0.9 * side / speed, (side - speed**2 / ramp_acceleration) / speed)
            if plateau < self.min_plateau_s:
                skipped.append(f"{math.degrees(speed):.0f}°/s ({max(plateau, 0.0):.2f} s)")
                continue
            far = context.start.copy()
            far[joint] += sign * (speed * plateau + speed**2 / ramp_acceleration)
            label = f"{self.display_label}: {math.degrees(speed):.1f}°/s"
            # WHY the kernels' trapezoid (private there): the plateau math stays the shared one the velocity sweep uses
            pieces.append(k._trapezoid(context.start, far, joint, speed, ramp_acceleration, context.rate_hz, label=f"{label} out"))
            pieces.append(k._trapezoid(far, context.start, joint, speed, ramp_acceleration, context.rate_hz, label=f"{label} back"))
            pieces.append(k.hold(context.start, self.dwell_s, context.rate_hz, label=f"{self.display_label}: between runs"))
        if skipped:
            context.note(f"{self.display_label}: skipped {', '.join(skipped)}: a plateau under {self.min_plateau_s:g} s "
                         "(the room is too short at that speed; the sines carry it)")
        if not pieces:
            raise ValueError("no speed has room for a plateau here; lower speed_fractions or min_plateau_s")
        return pieces
