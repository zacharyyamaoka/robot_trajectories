from __future__ import annotations

import numpy as np

from ..registry import LIMITS, kind
from ..segments import chain
from ..time_parameterization import TimeParameterization
from .cartesian_path import cartesian_path


@kind(use_cases=("cycle_time", "speed_limits"), needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
      unchecked=("collision_model", "reach_map"), catalog_id="adept_cycle", higher_order=True)
def adept_cycle(space, *, up_m=0.025, across_m=0.305, arrival_tolerance_m=0.00025, cycles=1, up=(0.0, 0.0, 1.0),
                across=(0.0, 1.0, 0.0), time_parameterization=TimeParameterization.minimum_jerk(peak_speed_fraction=1.0),
                eef_step_m=0.001, tolerance_m=0.0001, label="cycle", start=None):
    """The industry cycle-time benchmark: ``up_m`` up, ``across_m`` across, ``up_m`` down, then the same back, ``cycles`` times
    (the Adept cycle: 25 mm, 305 mm, 25 mm; "extended" is 25 / 700 / 25 mm). Six straight ``cartesian_path`` lines a cycle, each
    as fast as the joint limits allow (``time_parameterization`` defaults to full speed: the duration IS the result), a stop at each corner.

    ``up`` and ``across`` are directions in the base frame (a planar robot's plane: up (0, -1, 0) on the five-bar fixture).
    Every arrival must be within ``arrival_tolerance_m`` of its point (the benchmark's 0.25 mm, Schneider's number); a cycle
    that cannot is refused. The tool keeps its start orientation. WHY a stop at every corner: a blended Adept cycle does not stop,
    but a blend is a curve, which is ``cartesian_arc`` / ``via_point_spline``'s; the stops make this the conservative cycle time.
    A higher-order kind: ``cartesian_path`` lines, chained."""
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    up_axis, across_axis = np.asarray(up, float), np.asarray(across, float)
    for name, axis in (("up", up_axis), ("across", across_axis)):
        if np.linalg.norm(axis) < 1e-12:
            raise ValueError(f"{name} must be a nonzero direction")
    up_axis, across_axis = up_axis / np.linalg.norm(up_axis), across_axis / np.linalg.norm(across_axis)
    if tolerance_m > arrival_tolerance_m:
        raise ValueError(f"tolerance_m {tolerance_m} m (the inverse-kinematics tolerance) must not exceed arrival_tolerance_m "
                         f"{arrival_tolerance_m} m")
    origin = space.forward(start)[:3, 3]
    raised, far = origin + up_m * up_axis, origin + up_m * up_axis + across_m * across_axis
    landing = origin + across_m * across_axis
    corners = [("up", raised), ("across", far), ("down", landing), ("up", far), ("back across", raised), ("down", origin)]
    lines, here = [], start
    for cycle in range(1, cycles + 1):
        for name, point in corners:
            line = cartesian_path(space, [point], time_parameterization=time_parameterization, eef_step_m=eef_step_m,
                                  tolerance_m=tolerance_m, label=f"{label} {cycle}: {name}", start=here)
            miss = float(np.linalg.norm(space.forward(line.positions[-1])[:3, 3] - point))
            if miss > arrival_tolerance_m:
                raise ValueError(f"{label} {cycle}: {name}: arrives {1000 * miss:.3f} mm from its point, over the "
                                 f"{1000 * arrival_tolerance_m:.3f} mm arrival tolerance")
            lines.append(line)
            here = line
    return chain(*lines)
