from __future__ import annotations

import numpy as np

from ..registry import LIMITS, kind
from ..segments import Motion
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator
from . import _lane_b as b


@kind(use_cases=("gravity_id",), needs=(*LIMITS, "forward_kinematics", "dynamics_model"), unchecked=("collision_model",),
      catalog_id="quasi_static_pose_holds")
def quasi_static_pose_holds(space, *, poses=48, hold_s=2.0, dither_amplitude_rad=0.01, min_fold_margin=0.1, seed=20261113,
                            waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5), label="pose holds",
                            start=None):
    """A pose set: ``poses`` admitted configurations spread over the room, each held ``hold_s`` while every joint dithers slowly by
    +-``dither_amplitude_rad`` (a faded sine, a quarter period apart per joint) so Coulomb friction is crossed both ways and
    cancels from the averaged torque: masses, centres of mass and mount tilt by regression.

    A configuration is admitted when the kinematics has a tool pose there, nothing collides (when the space checks), the
    drive can hold it (the dynamics' holding torque inside the torque limits, when stated) and, where the kinematics reports a
    ``fold_margin(q)``, it clears ``min_fold_margin``. The stations (rad) and their holding torques are in the segment entry
    (``stations``, ``hold_torque_n_m``); the legs between them run under ``waypoint_generator``. Seeded: ``seed`` is part of the recipe.

    WHY the dither: holds on static friction over-fed the wrist (x1.22, the 11-14 lesson); the 15c fit used friction-cancelled
    moving rows."""
    if poses < 1 or hold_s <= 0 or dither_amplitude_rad < 0:
        raise ValueError(f"quasi_static_pose_holds needs poses >= 1, hold_s > 0, dither_amplitude_rad >= 0; got {poses}, {hold_s}, {dither_amplitude_rad}")

    def admit(q: np.ndarray) -> bool:
        if not (b.tool_ok(space, q) and b.fold_ok(space, q, min_fold_margin) and b.holdable(space, q)):
            return False
        if space.checks_collision and dither_amplitude_rad > 0:
            for joint in range(space.dof):
                for sign in (1.0, -1.0):
                    swung = q.copy()
                    swung[joint] += sign * dither_amplitude_rad
                    if space.in_collision(swung):
                        return False
        return True

    stations = b.draw_stations(space, start, poses, seed=seed, pad_rad=dither_amplitude_rad, admit=admit)
    pieces = b.station_pieces(space, start, stations, waypoint_generator, label=label,
                              dwell=lambda index, q: b.faded_dither(space, q, hold_s, dither_amplitude_rad, f"{label}: station {index}"))
    holding = [[float(v) for v in b.hold_effort(space, q)] for q in stations]
    entry = {**b.stations_entry(stations), "hold_torque_n_m": holding, "dither_amplitude_rad": float(dither_amplitude_rad)}
    notes = [f"{poses} admitted poses (a farthest-point pick from a seeded draw), each held {hold_s:g} s dithering +-{np.degrees(dither_amplitude_rad):.2f}°"]
    return Motion(pieces, notes, entry)
