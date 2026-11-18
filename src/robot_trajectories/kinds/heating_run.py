from __future__ import annotations

import numpy as np

from ..registry import LIMITS, kind
from ..segments import Motion
from . import _lane_b as b


@kind(use_cases=("thermal",), needs=(*LIMITS, "torque_limits"), catalog_id="heating_run", command="effort",
      until="the winding temperature reaches its limit (100 °C by default) or the board's (70 °C), or heat_s + cool_s have played")
def heating_run(space, joint=None, *, heat_current_fraction=0.5, probe_every_s=10.0, heat_s=600.0, cool_s=900.0,
                label="heating run", start=None):
    """Hold the rotor at standstill with a constant current at ``heat_current_fraction`` of the drive's limit for ``heat_s``,
    probing the winding (its resistance, read between heats) every ``probe_every_s``, then log the cool-down for ``cool_s``: the
    motor's thermal model.

    A current, not a motion: the effort is the constant (the torque-equivalent of a d-axis current: the rotor holds
    stepper-fashion, it produces none), the positions are the start held, not predicted. The heating is cut into one band per
    probe interval so each probe reads at a band's end; the probe times are in the segment entry. The player ends the run early on
    ``until`` (a temperature limit), so the run is a plan, not a promise to play 25 minutes."""
    if not 0 < heat_current_fraction <= 1 or probe_every_s <= 0 or heat_s <= 0 or cool_s < 0:
        raise ValueError(f"heating_run needs 0 < heat_current_fraction <= 1, probe_every_s > 0, heat_s > 0, cool_s >= 0; got "
                         f"{heat_current_fraction}, {probe_every_s}, {heat_s}, {cool_s}")
    j = space.joint_index(joint)
    level = heat_current_fraction * float(np.asarray(space.limits.max_effort, float)[j])
    pieces, probes, elapsed, index = [], [], 0.0, 0
    while elapsed < heat_s - 1e-9:
        length = min(probe_every_s, heat_s - elapsed)
        t = b.sample_times(length, space.rate_hz)
        effort = np.zeros((len(t), space.dof))
        effort[:, j] = level
        index += 1
        pieces.append(b.held_effort_piece(space, start, effort, t, f"{label}: heating {index}"))
        elapsed += float(t[-1])
        probes.append(round(elapsed, 6))
    if cool_s > 0:
        t = b.sample_times(cool_s, space.rate_hz)
        pieces.append(b.held_effort_piece(space, start, np.zeros((len(t), space.dof)), t, f"{label}: cool-down"))
    notes = [f"constant {level:.3f} N.m-equivalent ({heat_current_fraction:g} of the drive's limit) for {heat_s:g} s, then {cool_s:g} s with none; "
             "positions are the start held (the rotor holds), not a prediction"]
    return Motion(pieces, notes, {"heat_effort_n_m": float(level), "probe_times_s": probes, "cool_s": float(cool_s)})
