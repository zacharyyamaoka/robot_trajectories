from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..registry import LIMITS, kind
from ..segments import Motion
from . import _lane_b as b

MARGIN_S = 0.5


@kind(use_cases=("pid_tuning", "bandwidth"), needs=(*LIMITS, "torque_limits"), catalog_id="relay_feedback", command="effort",
      ends="anywhere", better_with={"dynamics_model": "the plan is the joint held at its start for max_duration_s; the limit cycle is not predicted"},
      until="limit cycle read: cycles relay periods (6 by default, 2 switches each) after the first switch, or max_duration_s with "
            "no switching (the relay torque is below the breakaway)")
def relay_feedback(space, joint=None, *, relay_torque_nm=0.2, hysteresis_rad=0.001, cycles=6, max_duration_s=10.0,
                   label="relay feedback", start=None):
    """Replace the controller by a relay: the torque that holds the start pose plus or minus ``relay_torque_nm`` on the sign of
    the measured error (with ``hysteresis_rad``), and read the limit cycle's amplitude and period: the ultimate gain
    ``Ku = 4 d / (pi a)`` and period a PID rule starts from.

    The relay follows the measurement, so the real run is the player's loop; this kind's trajectory is the nominal plan, the
    model's prediction of that loop (an ideal relay on the model, which a real loop's delay slows), ending ``MARGIN_S`` after
    the requested cycles. If the model says the relay never switches (its torque is below the breakaway) the plan runs
    ``max_duration_s`` and the notes say so."""
    if relay_torque_nm <= 0 or hysteresis_rad < 0 or cycles < 1 or max_duration_s <= 0:
        raise ValueError("relay_feedback needs relay_torque_nm > 0, hysteresis_rad >= 0, cycles >= 1, max_duration_s > 0; got "
                         f"{relay_torque_nm}, {hysteresis_rad}, {cycles}, {max_duration_s}")
    j = space.joint_index(joint)
    hold = b.hold_effort(space, start) if space.dynamics is not None else np.zeros(space.dof)
    headroom = b.torque_headroom(space, j, hold) if space.dynamics is not None else float(np.asarray(space.limits.max_effort, float)[j])
    if relay_torque_nm > headroom:
        raise ValueError(f"relay_torque_nm {relay_torque_nm:g} N.m is above what the drive can add on {space.joint_names[j]}: {headroom:.3f} N.m")
    entry = {"relay": {"torque_n_m": float(relay_torque_nm), "hysteresis_rad": float(hysteresis_rad), "cycles": int(cycles),
                       "setpoint_rad": float(start[j])}}
    samples = int(round(max_duration_s * space.rate_hz)) + 1
    if space.dynamics is None or not callable(getattr(space.dynamics, "forward", None)):
        piece = k.hold(start, max_duration_s, space.rate_hz, label=f"{label}: held while the relay runs")
        return Motion([piece], ["no dynamics model with forward(): the plan is the start pose held; the player's relay runs on the "
                                "measured error"], entry)
    state = {"u": 1.0, "switches": 0, "stop_at": None}

    def effort_at(index, q, qd):
        error = start[j] - q[j]  # setpoint - measurement
        if state["u"] > 0 and error < -hysteresis_rad:
            state["u"], state["switches"] = -1.0, state["switches"] + 1
        elif state["u"] < 0 and error > hysteresis_rad:
            state["u"], state["switches"] = 1.0, state["switches"] + 1
        tau = hold.copy()
        tau[j] += state["u"] * relay_torque_nm
        return tau

    def stop(index, q, qd):
        if state["stop_at"] is None and state["switches"] >= 2 * cycles:
            state["stop_at"] = index + int(round(MARGIN_S * space.rate_hz))
        return state["stop_at"] is not None and index >= state["stop_at"]

    q, qd, qdd, effort = b.predict_until(space, start, effort_at, samples, stop=stop)
    notes = []
    if state["switches"] < 2 * cycles:
        notes.append(f"the model never completes {cycles} relay cycles with +-{relay_torque_nm:g} N.m ({state['switches']} switches in "
                     f"{max_duration_s:g} s): the relay torque is below what moves the joint, so the plan runs the whole window")
    else:
        sign = np.sign(effort[:, j] - hold[j])
        flips = np.flatnonzero(np.diff(sign) != 0)
        amplitude = 0.5 * float(np.ptp(q[:, j]))
        notes.append(f"predicted limit cycle on the ideal relay: amplitude {amplitude:.5f} rad, {len(flips)} switches in {len(q) / space.rate_hz:.2f} s "
                     "(a real loop's delay sets the true period)")
        entry["relay"]["predicted_amplitude_rad"] = amplitude
    t = np.arange(len(q)) / space.rate_hz
    return Motion([b.effort_piece(space, t, effort, start, label, motion=(q, qd, qdd))], notes, entry)
