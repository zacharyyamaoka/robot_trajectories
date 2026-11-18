from __future__ import annotations

import math

from .. import kernels as k
from ..registry import kind


@kind(use_cases=("backlash_id", "repeatability"), catalog_id="small_reversals")
def small_reversals(space, joint=None, *, amplitude_rad=0.01, speed_rad_s=0.02, reversals=6, dwell_s=1.0, label="small reversals",
                    start=None):
    """Small slow back-and-forth moves about the held start pose, one joint, the others still: out to +amplitude, then
    ``reversals`` full swings between -amplitude and +amplitude (each a reversal of direction), a dwell at every turning point,
    and back to the start. The load sits on one flank, then the other: backlash and lost motion, read by an output-side camera
    or tracker (an external measurement, not an input here).

    Each leg is a minimum-jerk move whose peak speed is ``speed_rad_s`` (1.875 x distance / duration, kernels.MIN_JERK_PEAK_VELOCITY),
    so the reversals are as slow as asked and as smooth as the limits like."""
    if amplitude_rad <= 0 or speed_rad_s <= 0 or reversals < 1 or dwell_s < 0:
        raise ValueError(f"small_reversals needs amplitude_rad > 0, speed_rad_s > 0, reversals >= 1, dwell_s >= 0; got "
                         f"{amplitude_rad}, {speed_rad_s}, {reversals}, {dwell_s}")
    j = space.joint_index(joint)
    bounds = space.bounds
    room = float(min(start[j] - bounds.low[j], bounds.high[j] - start[j]))
    if amplitude_rad > room:
        raise ValueError(f"small_reversals: +-{math.degrees(amplitude_rad):.3f}° about {space.joint_names[j]}'s start is outside its "
                         f"limits (only {math.degrees(room):.3f}° of room each way)")
    speed = min(speed_rad_s, float(bounds.max_velocity[j]))
    pieces: list[k.Arrays] = []
    here = start.copy()

    def go(offset: float, name: str) -> None:
        nonlocal here
        target = start.copy()
        target[j] += offset
        pieces.append(k.min_jerk(here, target, bounds, space.rate_hz, duration_s=k.MIN_JERK_PEAK_VELOCITY * abs(offset - (here[j] - start[j])) / speed,
                                 label=f"{label}: {name}"))
        if dwell_s > 0:
            pieces.append(k.hold(target, dwell_s, space.rate_hz, label=f"{label}: dwell"))
        here = target

    go(amplitude_rad, "to +amplitude")
    sign = -1.0
    for index in range(1, reversals + 1):
        go(sign * amplitude_rad, f"reversal {index}")
        sign = -sign
    go(0.0, "back to the held pose")
    return pieces
