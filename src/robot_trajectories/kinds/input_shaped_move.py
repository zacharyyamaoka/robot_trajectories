from __future__ import annotations

import math

import numpy as np

from ..registry import LIMITS, kind
from ..segments import Motion
from ..time_parameterization import TimeParameterization
from ._lane_c import SHAPERS, shaped_leg, shaper_impulses


@kind(use_cases=("tracking_eval", "stop_accuracy", "cycle_time"), needs=(*LIMITS, "dynamics_model"), catalog_id="input_shaped_move")
def input_shaped_move(space, target=None, *, mode_hz=3.0, damping_ratio=0.05, shaper="ZVD",
                      time_parameterization=TimeParameterization.minimum_jerk(peak_speed_fraction=0.5), label="input-shaped move",
                      start=None):
    """A rest-to-rest move to ``target`` (absolute joint positions, rad) convolved with a ZV or ZVD impulse sequence tuned to a
    lightly damped mode, so the arm ends with no residual vibration at that mode (Singer and Seering, 1990).

    ``mode_hz`` is the mode's DAMPED frequency, one number or one per joint, and ``damping_ratio`` its damping ratio: both are
    arguments, never guessed (a dynamics model predicts torque, not a flexible mode). The shaper is the impulses ``[1, K]/(1+K)``
    half a damped period apart (ZV) or ``[1, 2K, K^2]/(1+K)^2`` (ZVD, less sensitive to a wrong frequency), K = exp(-zeta pi /
    sqrt(1 - zeta^2)). The move is ``time_parameterization``'s leg to the target with each joint's copies of it added at those delays; it ends
    a half (ZV) or whole (ZVD) damped period after the leg would, at rest. Positions stay between the start and the target and
    no speed or acceleration peak exceeds the unshaped leg's, so the limits hold by construction. ``target=None`` moves half way to
    the top of each joint's room.

    The dynamics model is read for what it predicts, the peak joint torque of the shaped move (in the notes); a model that cannot
    predict one is refused as the rung's absence is."""
    if shaper not in SHAPERS:
        raise ValueError(f"input_shaped_move.shaper is one of {SHAPERS}, got {shaper!r}")
    if not time_parameterization.built_in:
        raise ValueError("input_shaped_move shapes a built-in time parameterization's leg (minimum-jerk or trapezoid); a plugin "
                         "law has no closed-form profile")
    modes = np.broadcast_to(np.asarray(mode_hz, float), (space.dof,))
    if target is None:
        goal = start + 0.5 * (space.room(start).high - start)
    else:
        goal = np.asarray(target, float).reshape(-1)
        if goal.shape != (space.dof,):
            raise ValueError(f"target needs {space.dof} joint positions {space.joint_names}, got {goal.size}")
        bounds = space.bounds
        outside = np.flatnonzero((goal < bounds.low - 1e-12) | (goal > bounds.high + 1e-12))
        if len(outside):
            joint = int(outside[0])
            raise ValueError(f"target puts {space.joint_names[joint]} at {math.degrees(goal[joint]):.2f}°, outside its limits "
                             f"{math.degrees(bounds.low[joint]):.2f}..{math.degrees(bounds.high[joint]):.2f}°")
    impulses = [shaper_impulses(shaper, float(mode), damping_ratio) for mode in modes]
    leg = shaped_leg(start, goal, time_parameterization, space.limits, space.rate_hz, impulses, label=f"{label}: {shaper}")
    peak = np.max(np.abs([space.torque(q, qd, qdd) for q, qd, qdd in zip(leg.q[::10], leg.qd[::10], leg.qdd[::10], strict=True)]), axis=0)
    delay = max(float(times[-1]) for times, _ in impulses)
    return Motion([leg], [f"{label}: {shaper} for a {', '.join(f'{m:g}' for m in modes)} Hz mode at damping {damping_ratio:g}, "
                          f"{delay:.3f} s longer than the unshaped leg; predicted peak torque "
                          f"{', '.join(f'{v:.3g}' for v in peak)} N·m"])
