from __future__ import annotations

from .. import kernels as k
from ..registry import kind


@kind(use_cases=("pid_tuning", "bandwidth", "inertia_id"), catalog_id="log_chirp")
def chirp(space, joint=None, *, start_hz=0.2, end_hz=5.0, duration_s=20.0, logarithmic=True, fade_s=1.0, margin=0.8,
          label="chirp", start=None):
    """A sine swept from start_hz to end_hz on one joint at ONE amplitude for the whole sweep (the shared kernel): its
    frequency response and bandwidth.

    ``logarithmic``: equal time per octave (True) or per hertz. ``fade_s``: a raised-cosine fade in and out, so it starts and
    ends at rest. ``margin``: the tightest of room, velocity and acceleration the amplitude reaches.

    WHY it keeps the shared kernel, whose highest frequency sets the amplitude everywhere: the round's frozen scenario is this
    motion, sample for sample. ``linear_chirp`` carries the catalog's constant-speed rule (larger at low frequency)."""
    if start_hz <= 0 or end_hz <= 0:
        raise ValueError(f"chirp frequencies must be positive, got {start_hz} and {end_hz} Hz")
    return k.chirp(start, space.joint_index(joint), space.room(start), space.rate_hz, start_hz=start_hz, end_hz=end_hz,
                   duration_s=duration_s, logarithmic=logarithmic, fade_s=fade_s, margin=margin, label=label)
