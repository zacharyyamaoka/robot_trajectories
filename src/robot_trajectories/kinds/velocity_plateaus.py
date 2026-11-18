from __future__ import annotations

import math

from .. import kernels as k
from ..registry import kind
from ..segments import Motion


@kind(use_cases=("friction_id", "speed_limits", "state_coverage"), catalog_id="velocity_plateaus")
def velocity_plateaus(space, joint=None, *, speed_fractions=(0.025, 0.05, 0.1, 0.175, 0.25, 0.35), plateau_s=2.0,
                      min_plateau_s=0.8, first_direction="+", acceleration_fraction=0.8, dwell_s=0.5, label="velocity plateaus",
                      start=None):
    """Constant-speed plateaus out and back from the start at a ladder of speeds (fractions of the velocity limit), each held
    long enough to settle: friction against speed (Coulomb, viscous, Stribeck, breakaway). The rack schedule's ``plateaus``.

    Faithful to schedule.py: one speed per run, out and back, each plateau as long as the room allows (at most ``plateau_s``);
    a speed whose plateau would be shorter than ``min_plateau_s`` is skipped (the sines carry it) and the notes say so. WHY
    ramps where the rack steps the velocity: a velocity step is not inside an acceleration limit; the ramps take
    ``acceleration_fraction`` of it and their distance comes out of the tenth of the room the rack's rule leaves spare."""
    if first_direction not in ("+", "-"):
        raise ValueError(f"velocity_plateaus.first_direction is '+' or '-', got {first_direction!r}")
    joint = space.joint_index(joint)
    room = space.room(start)
    sign = 1.0 if first_direction == "+" else -1.0
    side = (room.high[joint] - start[joint]) if sign > 0 else (start[joint] - room.low[joint])
    ramp_acceleration = acceleration_fraction * float(space.bounds.max_acceleration[joint])
    pieces: list[k.Arrays] = []
    skipped = []
    for fraction in speed_fractions:
        speed = fraction * float(space.bounds.max_velocity[joint])
        # the rack's rule (plateau <= 0.9 x room / speed); the ramps then come out of the last tenth, and only if they do not
        # fit there does the plateau shorten
        plateau = min(plateau_s, 0.9 * side / speed, (side - speed**2 / ramp_acceleration) / speed)
        if plateau < min_plateau_s:
            skipped.append(f"{math.degrees(speed):.0f}°/s ({max(plateau, 0.0):.2f} s)")
            continue
        far = start.copy()
        far[joint] += sign * (speed * plateau + speed**2 / ramp_acceleration)
        name = f"{label}: {math.degrees(speed):.1f}°/s"
        # WHY the kernels' trapezoid (private there): the plateau math stays the shared one the velocity sweep uses
        pieces.append(k._trapezoid(start, far, joint, speed, ramp_acceleration, space.rate_hz, label=f"{name} out"))
        pieces.append(k._trapezoid(far, start, joint, speed, ramp_acceleration, space.rate_hz, label=f"{name} back"))
        pieces.append(k.hold(start, dwell_s, space.rate_hz, label=f"{label}: between runs"))
    if not pieces:
        raise ValueError("no speed has room for a plateau here; lower speed_fractions or min_plateau_s")
    notes = [f"{label}: skipped {', '.join(skipped)}: a plateau under {min_plateau_s:g} s (the room is too short at that speed; "
             "the sines carry it)"] if skipped else []
    return Motion(pieces, notes)
