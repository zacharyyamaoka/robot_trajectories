from __future__ import annotations

import math

import numpy as np

from ..registry import kind
from ._lane_e import plugin_law

GRID_POINTS = 200  # WHY 200: the catalog row's own default
# The default path, as fractions of the room around the start: out to half the room (joints alternating in sign), across to
# -0.4 of it on every joint, and back. WHY a default at all (the dispatcher, 2026-11-18): every kind is callable as f(space)
# alone (tests/test_selfcheck.py, the catalog previews); a loop that turns back exercises curvature, not only a line.
DEFAULT_FRACTIONS = ((0.5, "alternating"), (-0.4, "all"))
TOPPRA_PACKAGE = "robot-trajectories-toppra"


@kind(use_cases=("cycle_time", "speed_limits"), catalog_id="time_optimal_path")
def time_optimal_path(space, path=None, *, velocity_fraction=1.0, acceleration_fraction=1.0, effort_fraction=None,
                      grid_points=GRID_POINTS, label="time-optimal path", start=None):
    """The fastest timing along a fixed joint path under ``velocity_fraction`` / ``acceleration_fraction`` of the space's
    limits, rest to rest: TOPP-RA (Pham & Pham 2018) on a cubic spline through every point, ``grid_points`` its grid.

    ``effort_fraction``: None (the default) bounds velocity and acceleration only; a number in (0, 1] also keeps the space's
    dynamics model's torque within that fraction of ``max_effort`` at every sample (TOPP-RA's torque-limited mode: never
    faster than the kinematic timing). It needs ``space.with_dynamics(model)`` and a ``max_effort``, else ``MissingRung``.

    ``path``: M x J rad, M >= 2, in order (a flat list on a 1-DOF robot). ``None`` (the default): a loop from the start out to
    half the room (joints alternating in sign), across to -0.4 of it on every joint, and back to the start. The path is fixed
    and only its timing chosen (path-velocity decomposition); a path that does not begin at ``start`` gets the house transition
    into it. Every point must lie inside the position limits; the first one outside is refused by name. Timed by the
    ``toppra`` time law, a plugin (``robot-trajectories-toppra``): without it the kind is locked (``MissingRung``)."""
    # WHY built here, not as a default argument: a default is evaluated at import, before plugins load (KINDS_TODO lane E)
    options = {"acceleration_fraction": acceleration_fraction, "grid_points": int(grid_points)}
    if effort_fraction is not None:
        # WHY only when asked: the kinematic law's options (and so its recipe) stay what they were before the torque mode existed
        options["effort_fraction"] = float(effort_fraction)
    law = plugin_law("time_optimal_path", "toppra", TOPPRA_PACKAGE, peak_speed_fraction=velocity_fraction, options=options)
    if path is None:
        room = space.room(start)
        half = np.minimum(start - room.low, room.high - start)
        alternating = np.where(np.arange(space.dof) % 2, -1.0, 1.0)
        rows = np.stack([start, *(start + fraction * half * (alternating if pattern == "alternating" else 1.0)
                                  for fraction, pattern in DEFAULT_FRACTIONS), start])
    else:
        rows = np.asarray(path, float)
        rows = rows.reshape(-1, space.dof) if rows.size % space.dof == 0 else rows
    if rows.ndim != 2 or rows.shape[1] != space.dof or len(rows) < 2:
        raise ValueError(f"path must be M x {space.dof} joint configurations ({space.joint_names}) with M >= 2, "
                         f"got shape {np.shape(path)}")
    bounds = space.bounds
    for index, row in enumerate(rows):
        outside = np.flatnonzero((row < bounds.low - 1e-12) | (row > bounds.high + 1e-12))
        if len(outside):
            joint = int(outside[0])
            raise ValueError(f"path point {index} puts {space.joint_names[joint]} at {math.degrees(row[joint]):.2f}°, outside its "
                             f"limits {math.degrees(bounds.low[joint]):.2f}..{math.degrees(bounds.high[joint]):.2f}°")
    # WHY the space's dynamics handed on: the torque-limited mode reads the model; the kinematic mode is never handed it
    return law.along(rows, space.limits, space.rate_hz, label=label, dynamics=space.dynamics).arrays
