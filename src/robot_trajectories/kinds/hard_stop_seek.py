from __future__ import annotations

import math

import numpy as np

from .. import kernels as k
from ..registry import LIMITS, kind
from ..segments import Motion
from . import _lane_b as b


@kind(use_cases=("homing",), needs=(*LIMITS, "torque_limits"), catalog_id="hard_stop_seek", ends="anywhere",
      until="stall: |qd| < 0.01 rad/s for stall_s (0.2 s) while the setpoint leads by more than stall_lead_rad (0.02 rad), at the "
            "top torque cap; or max_travel_rad (0.5 rad) past the start; the stalled position defines the zero")
def hard_stop_seek(space, joint=None, *, speed_rad_s=0.1, cap_steps_nm=(0.5, 1.0, 2.0), stall_lead_rad=0.02, max_travel_rad=0.5,
                   touches=1, back_off_rad=0.03, direction=1.0, label="hard-stop seek", start=None):
    """One joint pushes toward its hard stop at a slow slewed setpoint (``speed_rad_s``) under a low torque cap that steps up
    through ``cap_steps_nm`` (a stall at a low cap may be gravity or friction: push harder before calling it the stop), the
    others held, until it stalls at the top cap; that angle defines the zero. ``touches`` > 1 backs off ``back_off_rad`` and presses
    again (N touches averaged). ``direction``: +1 toward the high side, -1 toward the low.

    The trajectory is the nominal plan: the slewed setpoint up to the expected stop plus the margin the position limit allows
    (the setpoint never leaves the joint's limits, so it ends at the nearer of ``max_travel_rad`` and the limit), declared
    ``ends="anywhere"`` with ``until`` the stall. The torque-cap schedule is a cap, not a command: it is in the segment
    entry (``seek``). Refused without the drive's torque limit, or with a cap above it."""
    if speed_rad_s <= 0 or not cap_steps_nm or list(cap_steps_nm) != sorted(cap_steps_nm) or min(cap_steps_nm) <= 0 \
            or stall_lead_rad <= 0 or max_travel_rad <= 0 or touches < 1 or back_off_rad < 0 or direction not in (1.0, -1.0, 1, -1):
        raise ValueError("hard_stop_seek needs speed_rad_s > 0, rising positive cap_steps_nm, stall_lead_rad > 0, max_travel_rad > 0, "
                         f"touches >= 1, back_off_rad >= 0, direction +-1; got {speed_rad_s}, {cap_steps_nm}, {stall_lead_rad}, "
                         f"{max_travel_rad}, {touches}, {back_off_rad}, {direction}")
    j = space.joint_index(joint)
    cap = float(np.asarray(space.limits.max_effort, float)[j])
    # WHY refused, not clipped: a seek presses a real stop; a cap the drive cannot give is a different test than the one asked for
    if max(cap_steps_nm) > cap + 1e-12:
        raise ValueError(f"cap_steps_nm {tuple(cap_steps_nm)} reaches {max(cap_steps_nm):g} N.m, above {space.joint_names[j]}'s torque limit {cap:g} N.m")
    bounds = space.bounds
    sign = float(direction)
    room = float(bounds.high[j] - start[j]) if sign > 0 else float(start[j] - bounds.low[j])
    travel = min(max_travel_rad, room)
    if travel <= 1e-6:
        raise ValueError(f"{space.joint_names[j]} has no room to seek on that side")
    notes = []
    if travel < max_travel_rad:
        notes.append(f"the setpoint stops at the joint's position limit, {math.degrees(max_travel_rad - travel):.2f}° short of max_travel_rad: "
                     "a stop past the limit is found by the player on its own, the plan cannot command beyond it")
    target = start.copy()
    target[j] += sign * travel
    pieces: list[k.Arrays] = [b.slew(space, start, target, j, speed_rad_s, f"{label}: seek, touch 1")]
    for touch in range(2, touches + 1):
        backed = target.copy()
        backed[j] -= sign * min(back_off_rad, travel)
        pieces.append(k.min_jerk(target, backed, bounds, space.rate_hz, speed=0.3, label=f"{label}: back off"))
        pieces.append(b.slew(space, backed, target, j, speed_rad_s, f"{label}: press again, touch {touch}"))
    entry = {"seek": {"joint": space.joint_names[j], "direction": sign, "speed_rad_s": float(speed_rad_s),
                      "cap_steps_nm": [float(c) for c in cap_steps_nm], "stall_lead_rad": float(stall_lead_rad),
                      "max_travel_rad": float(max_travel_rad), "touches": int(touches)}}
    return Motion(pieces, notes, entry)
