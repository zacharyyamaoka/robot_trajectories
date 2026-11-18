from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..registry import kind
from ..segments import Motion
from ._lane_a import deg


@kind(use_cases=("friction_id", "inertia_id", "state_coverage"), catalog_id="slow_plus_fast_sines")
def slow_plus_fast_sines(space, joint=None, *, slow_hz=0.15, fast_hz=2.5, slow_range_fraction=0.8, fast_to_slow_ratio=0.16,
                         duration_s=20.0, fade_s=1.0, label="slow sine with a fast ripple", start=None):
    """A slow large sine with a small fast sine on top, faded in and out: friction that depends on load while moving, and inertia
    separated from gravity (BAM's sin_sin; the 2026-10 note's F4, rescaled to the joint's range).

    ``slow_range_fraction``: the tightest of room, velocity and acceleration lands on this fraction of its limit (the rule of
    ``kernels._scaled``). ``fast_to_slow_ratio``: the ripple's amplitude as a fraction of the slow sine's.

    WHY the amplitudes are solved and not the note's 0.25 / 0.04 rad: the rack runner scales them to the joint's range
    (slow = 0.8 x the smaller side / 1.16, ripple = 0.16 of that); here the same ratio is kept and the limits (the ripple's
    acceleration is the usual binding one) are solved for, so the kind needs no joint it was not told about."""
    if slow_hz <= 0 or fast_hz <= 0 or duration_s <= 0 or fast_to_slow_ratio < 0:
        raise ValueError("slow_plus_fast_sines needs positive frequencies and duration and a ratio >= 0")
    index = space.joint_index(joint)
    t = k._samples(duration_s, space.rate_hz)
    total = t[-1]
    w, dw, ddw = k._fade(t, total, fade_s)
    slow, fast = 2 * np.pi * slow_hz, 2 * np.pi * fast_hz
    ratio = fast_to_slow_ratio
    x = np.sin(slow * t) + ratio * np.sin(fast * t)
    dx = slow * np.cos(slow * t) + ratio * fast * np.cos(fast * t)
    ddx = -(slow**2) * np.sin(slow * t) - ratio * fast**2 * np.sin(fast * t)
    shape = (x * w)[:, None], (dx * w + x * dw)[:, None], (ddx * w + 2 * dx * dw + x * ddw)[:, None]
    q, qd, qdd = k._scaled(np.asarray(start, float), [index], *shape, space.room(start), slow_range_fraction)
    gain = float(np.max(np.abs(q[:, index] - start[index]))) / float(np.max(np.abs(shape[0]))) if np.any(shape[0]) else 0.0
    note = (f"{label}: slow {deg(gain)} at {slow_hz:g} Hz, ripple {deg(gain * ratio)} at {fast_hz:g} Hz "
            f"(the tightest of room, velocity and acceleration at {slow_range_fraction:g} of its limit)")
    return Motion([k.Arrays(t, q, qd, qdd, label)], [note])
