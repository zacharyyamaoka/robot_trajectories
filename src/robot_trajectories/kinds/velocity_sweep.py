from __future__ import annotations

from .. import kernels as k
from ..registry import kind


@kind(use_cases=("friction_id", "gravity_id"), catalog_id="constant_speed_triangle")
def velocity_sweep(space, joint=None, *, speeds_fraction=(0.1, 0.3, 0.6), span_fraction=0.8, dwell_s=0.5, label="velocity sweep",
                   start=None):
    """Constant-speed passes across the range, up and down at several speeds, corners rounded at 0.8 x the acceleration
    limit: the friction curve and the gravity torque (the shared kernel). ``span_fraction``: how much of the room around the
    start each pass crosses.

    Catalog row ``constant_speed_triangle`` at several speeds (the rack's own friction runs, one speed out and back from the
    origin, are ``velocity_plateaus``)."""
    return k.velocity_sweep(start, space.joint_index(joint), space.room(start), space.rate_hz, speeds_fraction=speeds_fraction,
                            span_fraction=span_fraction, dwell_s=dwell_s, label=label)
