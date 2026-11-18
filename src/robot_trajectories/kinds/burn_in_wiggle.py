from __future__ import annotations

import math

import numpy as np

from .. import kernels as k
from ..registry import LIMITS, kind
from ..segments import Motion
from . import _lane_b as b


@kind(use_cases=("smoke_test", "endurance", "thermal"), needs=(*LIMITS, "home_pose"), unchecked=("collision_model",),
      catalog_id="burn_in_wiggle")
def burn_in_wiggle(space, joint=None, *, amplitude_rad=0.436, speeds_rad_s=(0.349, 0.698, 1.047), sweeps=2, label="burn-in wiggle",
                   start=None):
    """Each joint (``joint=None`` on a 1-DOF robot, or every joint in turn on an arm; or the one named) is walked to its named
    home, then swept about it one joint at a time at rising speeds with the others held: ``sweeps`` sweeps (+amplitude, -amplitude,
    home) at each of ``speeds_rad_s``. Every tick is logged by the player (current, bus voltage, temperature, faults): a pass is
    no faults and the joint having tracked its sweep.

    The amplitude is the largest the room around home gives (0.9 of it) when the asked one does not fit, and a speed above the
    joint's limit is cut to it; both said in the notes. The collision checker, when the space has one, is held to every sample by
    the guard (a pose that collides refuses the kind). The catalog's gripper cycles are not here: they are one more joint of the
    tool, a call of its own."""
    if amplitude_rad <= 0 or not speeds_rad_s or min(speeds_rad_s) <= 0 or sweeps < 1:
        raise ValueError(f"burn_in_wiggle needs amplitude_rad > 0, positive speeds_rad_s, sweeps >= 1; got {amplitude_rad}, {speeds_rad_s}, {sweeps}")
    home = space.home.copy()
    bounds = space.bounds
    joints = [space.joint_index(joint)] if joint is not None or space.dof == 1 else list(range(space.dof))
    room = space.room(home)
    pieces: list[k.Arrays] = []
    notes: list[str] = []
    if not np.allclose(start, home, atol=1e-12):
        pieces.append(k.min_jerk(start, home, bounds, space.rate_hz, speed=0.5, label=f"{label}: walk to home"))
    for j in joints:
        name = space.joint_names[j]
        fits = 0.9 * float(min(home[j] - room.low[j], room.high[j] - home[j]))
        swing = min(amplitude_rad, fits)
        if swing < amplitude_rad:
            notes.append(f"{name}: amplitude cut from {math.degrees(amplitude_rad):.1f}° to {math.degrees(swing):.1f}° (0.9 of the room around home)")
        high, low = home.copy(), home.copy()
        high[j] += swing
        low[j] -= swing
        for speed in speeds_rad_s:
            used = min(speed, float(bounds.max_velocity[j]))
            if used < speed:
                notes.append(f"{name}: {math.degrees(speed):.0f}°/s cut to the joint's {math.degrees(used):.0f}°/s limit")
            for sweep in range(1, sweeps + 1):
                text = f"{label}: {name} at {math.degrees(used):.0f}°/s, sweep {sweep}"
                pieces.append(b.slew(space, home, high, j, used, f"{text}, up"))
                pieces.append(b.slew(space, high, low, j, used, f"{text}, down"))
                pieces.append(b.slew(space, low, home, j, used, f"{text}, home"))
    return Motion(pieces, notes)
