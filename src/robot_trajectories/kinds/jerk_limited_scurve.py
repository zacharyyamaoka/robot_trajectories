from __future__ import annotations

import numpy as np

from ..registry import kind
from ..time_parameterization import legs_through
from ._lane_e import plugin_law

TIMED = ("joint_limits", "velocity_limits", "acceleration_limits", "jerk_limits")
RUCKIG_PACKAGES = "robot-trajectories-ruckig (Ruckig Community) or robot-trajectories-ruckig-pro (the local Pro build), one of the two"


@kind(use_cases=("tracking_eval", "cycle_time", "speed_limits"), needs=TIMED, catalog_id="jerk_limited_scurve")
def jerk_limited_scurve(space, joint=None, *, targets_fraction=(0.5, -0.5), velocity_fraction=1.0, acceleration_fraction=1.0,
                        jerk_fraction=1.0, synchronization="phase", dwell_s=0.3, back_to_start=True,
                        label="jerk-limited S-curve", start=None):
    """Rest-to-rest moves through waypoints given as fractions of the room, each leg a time-optimal jerk-limited S-curve (the
    double-S profile, Ruckig) under ``velocity_fraction`` / ``acceleration_fraction`` / ``jerk_fraction`` of the space's limits.

    ``synchronization``: ``"phase"`` (default) keeps a multi-joint leg a straight line in joint space (memory
    five-bar-regression, "Phase, not Time"); ``"time"``, ``"time_if_necessary"`` or ``"none"`` as Ruckig names them. Timed by
    the ``ruckig`` time parameterization, a plugin (``robot-trajectories-ruckig``, or the Pro drop-in): without one installed the kind is
    locked (``MissingRung``) naming the packages. ``joint_waypoints(space, ..., time_parameterization=TimeParameterization("ruckig", ...))`` takes absolute
    configurations instead."""
    for target in targets_fraction:
        values = target if isinstance(target, tuple) else (target,)
        if any(abs(value) > 1.0 for value in values):
            raise ValueError(f"jerk_limited_scurve.targets_fraction must lie in [-1, 1], got {target}")
    if not targets_fraction:
        raise ValueError("jerk_limited_scurve needs at least one waypoint in targets_fraction")
    # WHY built here, not as a default argument: a default is evaluated at import, before plugins load (KINDS_TODO lane E);
    # without a plugin serving "ruckig" the kind is locked (MissingPlugin, a MissingRung naming the package), never broken
    law = plugin_law("jerk_limited_scurve", "ruckig", RUCKIG_PACKAGES, peak_speed_fraction=velocity_fraction,
                     options={"acceleration_fraction": acceleration_fraction, "jerk_fraction": jerk_fraction,
                              "synchronization": synchronization})
    joints = space.joint_indices(joint)
    room = space.room(start)
    half = np.minimum(start - room.low, room.high - start)
    rows = []
    for target in targets_fraction:
        values = np.broadcast_to(np.asarray(target, float), (len(joints),))
        row = start.copy()
        row[joints] += values * half[joints]
        rows.append(row)
    if back_to_start:
        rows.append(start.copy())
    return legs_through(start, rows, law, space.limits, space.rate_hz, dwell_s=dwell_s, label=label, dynamics=space.dynamics)
