from __future__ import annotations

import numpy as np

from ..registry import LIMITS, kind
from ..segments import Motion
from . import _lane_b as b


@kind(use_cases=("backlash_id",), needs=(*LIMITS, "torque_limits"), catalog_id="locked_output_torque_ramp", command="effort")
def locked_output_torque_ramp(space, joint=None, *, peak_torque_nm=2.0, ramp_s=4.0, cycles=2, label="locked-output ramp", start=None):
    """With the output clamped in a fixture, ramp the motor torque 0 -> +peak -> 0 -> -peak -> 0, ``ramp_s`` per ramp, ``cycles``
    times, and read the motor angle: backlash and gear stiffness.

    The output is locked, so the positions are the start pose held (stated in the notes, not predicted) and the effort is the
    whole command; nothing else holds the pose, so no gravity torque is added. A peak above the drive's limit is refused."""
    if peak_torque_nm <= 0 or ramp_s <= 0 or cycles < 1:
        raise ValueError(f"locked_output_torque_ramp needs peak_torque_nm > 0, ramp_s > 0, cycles >= 1; got {peak_torque_nm}, {ramp_s}, {cycles}")
    j = space.joint_index(joint)
    cap = float(np.asarray(space.limits.max_effort, float)[j])
    if peak_torque_nm > cap + 1e-12:
        raise ValueError(f"peak_torque_nm {peak_torque_nm:g} N.m is above {space.joint_names[j]}'s torque limit {cap:g} N.m")
    per = int(round(ramp_s * space.rate_hz))
    up = np.linspace(0.0, 1.0, per + 1)
    one_cycle = np.concatenate([up[:-1], up[::-1][:-1], -up[:-1], -up[::-1][:-1]])
    shape = np.append(np.tile(one_cycle, cycles), 0.0)
    t = np.arange(len(shape)) / space.rate_hz
    effort = np.zeros((len(t), space.dof))
    effort[:, j] = peak_torque_nm * shape
    notes = [f"the output is locked: the positions are {space.joint_names[j]}'s start held, not a prediction; the motor angle is read "
             "from the drive"]
    return Motion([b.held_effort_piece(space, start, effort, t, label)], notes,
                  {"peak_torque_n_m": float(peak_torque_nm), "ramp_s": float(ramp_s), "cycles": int(cycles), "output": "locked"})
