from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..registry import LIMITS, kind
from ..segments import Motion
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator
from . import _lane_b as b


@kind(use_cases=("gravity_id", "smoke_test"), needs=(*LIMITS, "torque_limits", "forward_kinematics", "dynamics_model"),
      catalog_id="weightless_float", command="effort", ends="anywhere",
      until="the last float has run (float_s, 5 s by default) or the drift passes the player's guard; each earlier float ends "
            "when float_s has run")
def weightless_float(space, *, float_s=5.0, hold_s=1.0, poses=5, seed=11,
                     waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5), label="weightless float",
                     start=None):
    """At each of ``poses`` admitted, holdable configurations spread over the room: hold under the position loop for ``hold_s``,
    then drop kp and kd to zero and keep sending the model's gravity torque for ``float_s``. The drift over the float is how well
    gravity is modelled ("it feels weightless", made measurable).

    The effort is the gravity torque at the pose the model predicts the joints to be in (what the host recomputes each tick);
    under the same model the prediction is therefore no drift, and the measured drift is the model's error. It ends in a float
    (``ends="anywhere"``). A pose is admitted when the kinematics has a tool pose there, nothing collides (when checked) and the
    drive can hold it. The stations are in the segment entry."""
    if float_s <= 0 or hold_s <= 0 or poses < 1:
        raise ValueError(f"weightless_float needs float_s > 0, hold_s > 0, poses >= 1; got {float_s}, {hold_s}, {poses}")
    stations = b.draw_stations(space, start, poses, seed=seed, admit=lambda q: b.tool_ok(space, q) and b.holdable(space, q))
    pieces: list[k.Arrays] = []
    here = np.asarray(start, float)
    samples = int(round(float_s * space.rate_hz)) + 1
    for index, q in enumerate(stations, start=1):
        if not np.allclose(here, q, atol=1e-12):
            pieces.append(waypoint_generator.leg(here, q, space.limits, space.rate_hz,
                                                    label=f"{label}: move to station {index}", dynamics=space.dynamics))
        pieces.append(k.hold(q, hold_s, space.rate_hz, label=f"{label}: station {index} held"))
        swing = b.predict_until(space, q, lambda i, position, velocity: b.hold_effort(space, position), samples)
        pieces.append(b.Piece(np.arange(len(swing[0])) / space.rate_hz, swing[0], swing[1], swing[2],
                              f"{label}: station {index} floating", effort=swing[3]))
        here = swing[0][-1]
    notes = [f"{poses} poses, each held {hold_s:g} s then floated {float_s:g} s on the model's gravity torque alone"]
    return Motion(pieces, notes, b.stations_entry(stations))
