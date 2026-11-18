from __future__ import annotations

from .. import kernels as k
from ..registry import kind


@kind(use_cases=("bandwidth",), catalog_id="faded_sine")
def sine_dwell(space, joint=None, *, frequency_hz=1.0, duration_s=10.0, fade_s=1.0, margin=0.8, label="sine dwell", start=None):
    """One sine at a fixed frequency on one joint, faded in and out: the gain and phase at that one frequency.

    Dwelling at one frequency averages the noise out of that point, where a chirp passes through it once."""
    return k.chirp(start, space.joint_index(joint), space.room(start), space.rate_hz, start_hz=frequency_hz, end_hz=frequency_hz,
                   duration_s=duration_s, logarithmic=False, fade_s=fade_s, margin=margin, label=label)
