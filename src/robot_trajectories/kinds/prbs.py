from __future__ import annotations

import numpy as np

from ..registry import kind
from ..segments import Motion
from ._lane_a import deg, level_sequence, m_sequence


@kind(use_cases=("bandwidth", "delay_id", "inertia_id"), catalog_id="prbs")
def prbs(space, joint=None, *, register_stages=7, clock_period_s=0.05, amplitude_rad=0.05, periods=2, edge_fraction=0.5, margin=0.8,
         seed=0, label="prbs", start=None):
    """A maximum-length binary sequence (period 2^register_stages - 1 clock periods) of two levels, +- ``amplitude_rad`` about the
    start, each held one clock period, every change a minimum-jerk edge so a position loop can follow it. A flat spectrum up to
    about 0.44 / clock period: bandwidth, delay and inertia in one run. It starts at the start, goes to its first level, and ends
    back at rest at the start.

    ``seed`` picks the register's starting fill (a non-zero state), so it picks where in the period the sequence begins.
    ``edge_fraction``: an edge takes this fraction of a clock period.

    WHY the amplitude gives way and the clock does not: the clock sets the spectrum; the edges must fit the velocity and
    acceleration limits (and the room), so when they do not the amplitude is lowered and a note says by how much. Binary position
    steps are harsh on a position-controlled joint (drives inject a PRBS on the velocity or current loop), which is why the edge
    is a minimum-jerk one and not a step."""
    if periods < 1 or amplitude_rad <= 0 or clock_period_s <= 0:
        raise ValueError("prbs needs periods >= 1, amplitude_rad > 0 and clock_period_s > 0")
    index = space.joint_index(joint)
    bits = np.tile(m_sequence(int(register_stages), int(seed)), int(periods))
    arrays, gain = level_sequence(start, index, space.room(start), np.where(bits == 1, 1.0, -1.0) * amplitude_rad,
                                  [clock_period_s] * len(bits), space.rate_hz, edge_fraction=edge_fraction, margin=margin, label=label)
    note = f"{label}: {deg(gain * amplitude_rad)} levels, {2**register_stages - 1} clock periods of {clock_period_s:g} s x {periods}"
    if gain < 1.0:
        note += f" (asked {deg(amplitude_rad)}: lowered so every edge fits the velocity and acceleration limits and the room)"
    return Motion([arrays], [note])
