from __future__ import annotations

from .. import kernels as k
from ..registry import kind


@kind(use_cases=("stop_accuracy", "pid_tuning", "repeatability", "smoke_test"), needs=("joint_limits",), catalog_id="hold_at_pose")
def hold(space, duration_s=1.0, *, settle_window_s=0.3, label="hold", start=None):
    """Hold where it is and read the settled stop from the hold's final window: the segment that joins two moves, and the
    stop-accuracy reading itself (the catalog's hold_s and settle window)."""
    if duration_s < 0 or settle_window_s < 0:
        raise ValueError(f"hold needs duration_s and settle_window_s >= 0, got {duration_s}, {settle_window_s}")
    window = min(settle_window_s, duration_s)
    settle = duration_s - window
    # WHY two pieces: the read window shows as its own band, so a scorer reads the stop where the catalog says to.
    if window <= 0 or settle * space.rate_hz < 2:
        return [k.hold(start, duration_s, space.rate_hz, label=label)]
    return [k.hold(start, settle, space.rate_hz, label=label),
            k.hold(start, window, space.rate_hz, label=f"{label}: settled-stop window")]
