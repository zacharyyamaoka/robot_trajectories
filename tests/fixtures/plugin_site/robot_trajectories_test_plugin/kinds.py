"""A kind from outside the package: registered by its entry point, listed and guarded like a built-in one."""

from __future__ import annotations

import numpy as np

from robot_trajectories.registry import kind
from robot_trajectories.time_parameterization import TimeParameterization, legs_through


@kind(use_cases=("demo", "smoke_test"), family="point_to_point", starts_as="path", dimensions=("1D", "2D", "3D", "6D"))
def smoothstep_moves(space, joint=None, *, targets_fraction=(0.4, -0.4), peak_speed_fraction=0.5, dwell_s=0.2,
                     label="smoothstep moves", start=None):
    """Moves to fractions of the room and back, each leg timed by the plugin's own smoothstep time parameterization."""
    law = TimeParameterization("smoothstep_test", peak_speed_fraction=peak_speed_fraction)
    joints = space.joint_indices(joint)
    room = space.room(start)
    half = np.minimum(start - room.low, room.high - start)
    rows = []
    for fraction in (*targets_fraction, 0.0):
        row = start.copy()
        row[joints] += fraction * half[joints]
        rows.append(row)
    return legs_through(start, rows, law, space.limits, space.rate_hz, dwell_s=dwell_s, label=label)
