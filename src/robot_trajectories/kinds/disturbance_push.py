from __future__ import annotations

from .. import kernels as k
from ..registry import kind
from ..segments import Motion


@kind(use_cases=("disturbance_rejection", "pid_tuning"), needs=("joint_limits", "home_pose"), catalog_id="disturbance_push")
def disturbance_push(space, *, push_n=15.0, push_s=0.4, push_at_s=0.5, push_direction=(1.0, 0.0, 0.0), duration_s=2.0,
                     label="hold under a push", start=None):
    """A plain hold of the start pose (home by default) for ``duration_s``; the generated part is the disturbance schedule beside
    it, in the segment entry: a force of ``push_n`` newtons along ``push_direction`` (the base frame's x by default) on the tool
    from ``push_at_s`` for ``push_s`` seconds (a simulator applies it; on the rig a person pushes at the cue). Recovery time and
    overshoot are read from the hold after it.

    The hold is cut into three bands (before, during, after the push) so a scorer reads each; the push must end before the hold
    does, leaving a recovery window."""
    if push_n <= 0 or push_s <= 0 or push_at_s < 0 or duration_s <= 0 or len(push_direction) != 3:
        raise ValueError(f"disturbance_push needs push_n > 0, push_s > 0, push_at_s >= 0, duration_s > 0 and a 3-vector direction; got "
                         f"{push_n}, {push_s}, {push_at_s}, {duration_s}, {push_direction}")
    if push_at_s + push_s >= duration_s:
        raise ValueError(f"the push (from {push_at_s:g} s for {push_s:g} s) must end before the {duration_s:g} s hold does: no window to recover in")
    pieces = [k.hold(start, push_at_s, space.rate_hz, label=f"{label}: before the push")] if push_at_s * space.rate_hz >= 2 else []
    pieces.append(k.hold(start, push_s, space.rate_hz, label=f"{label}: during the push"))
    after = duration_s - push_at_s - push_s
    pieces.append(k.hold(start, after, space.rate_hz, label=f"{label}: recovery"))
    entry = {"push": {"force_n": float(push_n), "direction": [float(v) for v in push_direction], "frame": "base", "start_s": float(push_at_s),
                      "duration_s": float(push_s)}}
    return Motion(pieces, [f"the trajectory is the hold; the {push_n:g} N push from {push_at_s:g} s for {push_s:g} s is the entry's "
                            "'push' schedule"], entry)
