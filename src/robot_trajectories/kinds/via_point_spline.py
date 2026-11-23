from __future__ import annotations

import math

import numpy as np

from .. import kernels as k
from ..registry import LIMITS, kind
from ..segments import Motion
from ..time_parameterization import TimeParameterization
from ._lane_c import blend_polyline, natural_spline, uniform_arc

INTERPOLATIONS = ("blend", "spline")


def _joint_radius(space, corner: np.ndarray, neighbour: np.ndarray, metres: float) -> float:
    """The joint-space distance from ``corner`` toward ``neighbour`` at which the tool is ``metres`` from the tool at the corner
    (bisection on the fraction of the side; the whole side when the tool never gets that far)."""
    length = float(np.linalg.norm(neighbour - corner))
    reference = space.forward(corner)[:3, 3]

    def away(fraction: float) -> float:
        return float(np.linalg.norm(space.forward(corner + fraction * (neighbour - corner))[:3, 3] - reference))

    if away(1.0) <= metres:
        return length
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if away(middle) < metres else (low, middle)
    return low * length


def _peak_jerk(arrays: k.Arrays, rate_hz: float) -> np.ndarray:
    return np.max(np.abs(np.diff(arrays.qdd, axis=0)), axis=0) * rate_hz


@kind(use_cases=("tracking_eval", "cycle_time", "demo"), needs=(*LIMITS, "jerk_limits"),
      better_with={"jerk_limits": "timed by the speed and acceleration limits alone: the jerk is not held to a limit"},
      unchecked=("collision_model",), catalog_id="via_point_spline")
def via_point_spline(space, via_points=None, *, interpolation="blend", blend_radius_rad=0.1, blend_radius_m=None, stop_at=(),
                     time_parameterization=TimeParameterization.minimum_jerk(peak_speed_fraction=0.5), dwell_s=0.0,
                     label="via-point spline", start=None):
    """One smooth motion through ``via_points`` (M x J absolute joint positions, rad) without stopping at them, at rest at the start
    and at the last one: the blended sequence of Pilz (a corner is cut, never stopped at), or a spline through the points.

    ``interpolation="blend"``: straight joint-space sides, each inner corner cut by a smooth curve (a quintic Bezier with zero end
    curvature, so the path's tangent and curvature are continuous) that starts and ends ``blend_radius_rad`` from the corner along
    each side (never more than half a side). With a kinematics in the space, ``blend_radius_m`` states it Pilz's way instead: the
    tool metres from the corner. The path passes near each via point, within the blend radius, not through it.
    ``interpolation="spline"``: a natural cubic spline in the chord length, through every via point exactly (it can overshoot
    between them; one that leaves a joint's range is refused naming the joint).

    ``stop_at``: indices of via points to stop at (rest, then ``dwell_s``); the path is cut there into rest-to-rest runs. One
    run is timed along its whole length by ``time_parameterization``. When the space states a jerk limit the run is slowed until its peak jerk is
    inside it (a trapezoid has unlimited jerk and is refused then); without one the jerk is not limited."""
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"via_point_spline.interpolation is one of {INTERPOLATIONS}, got {interpolation!r}")
    if via_points is None:  # WHY: a deterministic default, so the kind can be called as f(space): three points at fractions of the room
        room = space.room(start)
        via_points = [start + f * (room.high - start) for f in (0.3, 0.1)] + [start + 0.3 * (room.low - start)]
    rows = np.asarray(via_points, float)
    rows = rows.reshape(-1, space.dof) if rows.size % space.dof == 0 else rows
    if rows.ndim != 2 or rows.shape[1] != space.dof or len(rows) == 0:
        raise ValueError(f"via_points must be M x {space.dof} joint positions ({space.joint_names}), got shape {np.shape(via_points)}")
    bounds = space.bounds
    for index, row in enumerate(rows):
        outside = np.flatnonzero((row < bounds.low - 1e-12) | (row > bounds.high + 1e-12))
        if len(outside):
            joint = int(outside[0])
            raise ValueError(f"via point {index} puts {space.joint_names[joint]} at {math.degrees(row[joint]):.2f}°, outside its limits "
                             f"{math.degrees(bounds.low[joint]):.2f}..{math.degrees(bounds.high[joint]):.2f}°")
    points = np.vstack([start, rows])
    for index in range(1, len(points)):
        if float(np.linalg.norm(points[index] - points[index - 1])) < 1e-9:
            raise ValueError(f"via_point_spline: via point {index - 1} is where the previous one (or the start) is; "
                             "drop the repeat (a pause is dwell_s with stop_at)")
    stops = sorted({int(i) for i in stop_at})
    if any(not 0 <= i < len(rows) for i in stops):
        raise ValueError(f"stop_at holds via-point indices 0..{len(rows) - 1}, got {tuple(stop_at)}")
    cuts = sorted({*(i + 1 for i in stops), len(points) - 1})  # indices into points where a run ends (the last always)
    if blend_radius_m is not None and not space.has("forward_kinematics"):
        raise ValueError("via_point_spline.blend_radius_m is the tool's distance: it needs kinematics (space.with_kinematics); "
                         "blend_radius_rad states the blend in joint space")
    if blend_radius_m is None and blend_radius_rad <= 0:
        raise ValueError(f"via_point_spline.blend_radius_rad must be positive, got {blend_radius_rad}")
    jerk_limit = None
    if space.has("jerk_limits"):
        jerk_limit = np.asarray(space.limits.max_jerk, float)
        if time_parameterization.shape == "trapezoid":
            raise ValueError("a trapezoid time parameterization has unlimited jerk but this space states a jerk limit: use "
                             "TimeParameterization.minimum_jerk")
    pieces: list[k.Arrays] = []
    notes: list[str] = []
    first = 0
    for run_index, last in enumerate(cuts, start=1):
        run = points[first:last + 1]
        name = f"{label}: run {run_index}" if len(cuts) > 1 else label
        if len(run) == 2:
            path, grid = run, np.array([0.0, 1.0])
        elif interpolation == "spline":
            path = uniform_arc(natural_spline(run)[0])
            grid = np.linspace(0.0, 1.0, len(path))
            over = np.flatnonzero(np.any((path < bounds.low - 1e-9) | (path > bounds.high + 1e-9), axis=0))
            if len(over):
                joint = int(over[0])
                raise ValueError(f"the spline leaves {space.joint_names[joint]}'s range ({math.degrees(path[:, joint].min()):.2f}.."
                                 f"{math.degrees(path[:, joint].max()):.2f}° against {math.degrees(bounds.low[joint]):.2f}.."
                                 f"{math.degrees(bounds.high[joint]):.2f}°) between via points: use interpolation='blend'")
        else:
            radii = np.zeros(len(run))
            for index in range(1, len(run) - 1):
                before, corner, after = run[index - 1], run[index], run[index + 1]
                if blend_radius_m is None:
                    wanted = blend_radius_rad
                else:  # WHY the smaller of the two sides: one symmetric blend, so the path stays tangent to both sides
                    wanted = min(_joint_radius(space, corner, before, blend_radius_m), _joint_radius(space, corner, after, blend_radius_m))
                radii[index] = min(wanted, 0.5 * float(np.linalg.norm(before - corner)), 0.5 * float(np.linalg.norm(after - corner)))
            path = uniform_arc(blend_polyline(run, radii))
            grid = np.linspace(0.0, 1.0, len(path))
        law = time_parameterization
        for _ in range(20):
            timed = law.along(path, space.limits, space.rate_hz, label=name, progress_grid=grid, dynamics=space.dynamics)
            # WHY the output is read again, the way the guard reads it (differences of the sampled velocity): a time parameterization reads a
            # path's slope and curvature on its grid, and where the path turns tightly (a blend radius small against the legs)
            # that differs from the sampled motion by a few percent; the run is slowed until the sampled motion fits
            accel = float(np.max(np.abs(np.diff(timed.arrays.qd, axis=0)) * space.rate_hz / bounds.max_acceleration))
            jerk = 0.0 if jerk_limit is None else float(np.max(_peak_jerk(timed.arrays, space.rate_hz) / jerk_limit))
            # WHY the cube root of the jerk ratio: scaling a motion's time by f scales its jerk by 1 / f^3 (its acceleration by 1 / f^2)
            factor = max(accel**0.5 if accel > 1.0 else 1.0, jerk ** (1.0 / 3.0) if jerk > 1.0 else 1.0)
            if factor <= 1.0:
                break
            # WHY the options carried over: a plugin law's settings (TOPP-RA's effort_fraction) must survive the slowing
            law = TimeParameterization(time_parameterization.shape, move_s=float(timed.arrays.t[-1]) * factor * 1.01,
                                       ramp_fraction=time_parameterization.ramp_fraction,
                                       min_move_s=time_parameterization.min_move_s, options=time_parameterization.options)
        else:
            raise ValueError("via_point_spline: no timing keeps this path inside the acceleration and jerk limits in 20 slowings")
        pieces.append(timed.arrays)
        if jerk_limit is not None:
            notes.append(f"{name}: peak jerk {100 * jerk:.0f} % of its limit")
        if dwell_s > 0 and last - 1 in stops:
            pieces.append(k.hold(run[-1], dwell_s, space.rate_hz, label=f"{label}: dwell"))
        first = last
    notes.append(f"{label}: {interpolation} through {len(rows)} via points, stopping at "
                 f"{'the last' if not stops else 'via ' + ', '.join(map(str, stops)) + ' and the last'}")
    return Motion(pieces, notes)
