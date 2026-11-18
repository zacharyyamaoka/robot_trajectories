from __future__ import annotations

import numpy as np

from ..registry import LIMITS, kind
from ..segments import Motion
from . import _lane_b as b

MARGIN_S = 0.2


@kind(use_cases=("friction_id", "backlash_id"), needs=(*LIMITS, "torque_limits"), catalog_id="breakaway_torque_ramp", command="effort",
      ends="anywhere", better_with={"dynamics_model": "the plan is the start pose held while the ramp runs; the breakaway is not predicted"},
      until="breakaway: |q - q_start| > leave_threshold_rad (0.002 rad by default), then margin 0.2 s; or the ramp reaches the "
            "drive's torque limit")
def breakaway_torque_ramp(space, joint=None, *, ramp_rate_nm_s=0.5, start_torque_nm=0.0, leave_threshold_rad=0.002, direction=1.0,
                          label="breakaway ramp", start=None):
    """Hold one flank, then ramp the torque on top of the torque that holds the pose, at ``ramp_rate_nm_s`` from
    ``start_torque_nm``, until the output leaves by ``leave_threshold_rad``: static friction in one direction
    (``direction`` +1 or -1; the other flank is a second call from the measured pose, the player reads the torque at the leave).

    The plan is the ramp up to the model's predicted leave plus a margin (up to the drive's torque limit if it predicts none);
    with no dynamics model the positions are held and only the ramp is the plan."""
    if ramp_rate_nm_s <= 0 or start_torque_nm < 0 or leave_threshold_rad <= 0 or direction not in (1.0, -1.0, 1, -1):
        raise ValueError("breakaway_torque_ramp needs ramp_rate_nm_s > 0, start_torque_nm >= 0, leave_threshold_rad > 0, "
                         f"direction +-1; got {ramp_rate_nm_s}, {start_torque_nm}, {leave_threshold_rad}, {direction}")
    j = space.joint_index(joint)
    modelled = space.dynamics is not None
    hold = b.hold_effort(space, start) if modelled else np.zeros(space.dof)
    cap = float(np.asarray(space.limits.max_effort, float)[j])
    ceiling = (b.torque_headroom(space, j, hold) if modelled else cap) - 1e-9
    if start_torque_nm >= ceiling:
        raise ValueError(f"start_torque_nm {start_torque_nm:g} N.m is not below what the drive can add: {ceiling:.3f} N.m")
    ramp_s = (ceiling - start_torque_nm) / ramp_rate_nm_s
    samples = int(np.floor(ramp_s * space.rate_hz)) + 1
    sign = float(direction)
    entry = {"ramp": {"rate_n_m_s": float(ramp_rate_nm_s), "start_n_m": float(start_torque_nm), "direction": sign,
                      "leave_threshold_rad": float(leave_threshold_rad), "ceiling_n_m": float(ceiling)}}

    def ramp(index):
        return start_torque_nm + ramp_rate_nm_s * index / space.rate_hz

    if not modelled or not callable(getattr(space.dynamics, "forward", None)):
        t = np.arange(samples) / space.rate_hz
        effort = np.tile(hold, (len(t), 1))
        effort[:, j] += sign * (start_torque_nm + ramp_rate_nm_s * t)
        piece = b.held_effort_piece(space, start, effort, t, f"{label}: torque ramp")
        return Motion([piece], ["no dynamics model with forward(): the positions are the start pose held; the effort is the ramp "
                                "(on top of whatever holds the pose)"], entry)
    state = {"left": None}

    def effort_at(index, q, qd):
        tau = hold.copy()
        tau[j] += sign * ramp(index)
        return tau

    def stop(index, q, qd):
        if state["left"] is None and abs(q[j] - start[j]) > leave_threshold_rad:
            state["left"] = index
        return state["left"] is not None and index >= state["left"] + int(round(MARGIN_S * space.rate_hz))

    q, qd, qdd, effort = b.predict_until(space, start, effort_at, samples, stop=stop)
    if state["left"] is None:
        notes = [f"the model predicts no breakaway below the drive's limit ({ceiling:.3f} N.m): the plan is the whole ramp"]
    else:
        torque = ramp(state["left"])
        notes = [f"predicted breakaway at {torque:.3f} N.m above the hold ({state['left'] / space.rate_hz:.2f} s into the ramp); the "
                 f"plan runs {MARGIN_S:g} s past it"]
        entry["ramp"]["predicted_breakaway_n_m"] = float(torque)
    t = np.arange(len(q)) / space.rate_hz
    return Motion([b.effort_piece(space, t, effort, start, label, motion=(q, qd, qdd))], notes, entry)
