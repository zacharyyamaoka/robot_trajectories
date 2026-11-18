from __future__ import annotations

from .. import kernels as k
from ..registry import kind


@kind(use_cases=(), family="point_to_point", starts_as="path", dimensions=("1D", "2D", "3D", "6D"))
def home(space, *, speed=0.5, label="home", start=None):
    """Move back to the space's home: the building block that ends a session where it began (no catalog row).

    ``speed``: the fraction of the velocity and acceleration limits the move may use."""
    return k.min_jerk(start, space.home, space.bounds, space.rate_hz, speed=speed, label=label)
