from __future__ import annotations

from .. import kernels as k
from ..registry import kind


@kind(use_cases=("full_dynamics_id", "bandwidth"), catalog_id="multisine")
def multisine(space, joints=None, *, band_hz=(0.2, 3.0), period_s=10.0, periods=2, seed=0, fade_s=1.0, margin=0.8,
              label="multisine", start=None):
    """Schroeder-phased sines on every joint at once (``joints``, default all), each joint on its own lines: coupled dynamics
    in one run. ``period_s`` sets the frequency resolution (1 / period_s); ``seed`` the phase offsets."""
    return k.multisine(start, space.joint_indices(joints), space.room(start), space.rate_hz, band_hz=band_hz, period_s=period_s,
                       periods=periods, seed=seed, fade_s=fade_s, margin=margin, label=label)
