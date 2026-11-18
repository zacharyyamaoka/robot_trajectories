from __future__ import annotations

import numpy as np

from ..registry import kind
from ..segments import Motion
from ._lane_a import deg, level_sequence


def _units(pattern: str) -> tuple[int, ...]:
    """``"doublet"`` is 1-1; otherwise dash-separated whole numbers of time units (``"3-2-1-1"``)."""
    if pattern == "doublet":
        return (1, 1)
    try:
        units = tuple(int(part) for part in pattern.split("-"))
    except ValueError:
        units = ()
    if len(units) < 2 or min(units) < 1:
        raise ValueError(f"doublet_3211.pattern is 'doublet' or whole time units joined by dashes (e.g. '3-2-1-1'), got {pattern!r}")
    return units


@kind(use_cases=("bandwidth", "inertia_id", "pid_tuning"), catalog_id="doublet_3211")
def doublet_3211(space, joint=None, *, unit_time_s=0.2, amplitude_rad=0.05, pattern="3-2-1-1", edge_fraction=0.5, margin=0.8,
                 label="doublet pulses", start=None):
    """A doublet (+A then -A) or the 3-2-1-1 train (3, 2, 1, 1 time units of alternating sign), as position pulses about the start:
    a short input with a broad, tunable spectrum, for where arm time is expensive (a flight-test input). It goes to its first
    level, runs the pulses, and ends back at rest at the start.

    ``pattern``: ``"doublet"``, ``"3-2-1-1"`` or any dash-separated whole units. ``edge_fraction``: an edge takes this fraction of
    the shortest pulse.

    WHY the amplitude gives way (as in ``prbs``): the pulse widths are the spectrum; the edges must fit the velocity and
    acceleration limits and the room, so the amplitude is lowered, with a note, when they do not."""
    if unit_time_s <= 0 or amplitude_rad <= 0:
        raise ValueError("doublet_3211 needs unit_time_s > 0 and amplitude_rad > 0")
    units = _units(pattern)
    index = space.joint_index(joint)
    offsets = amplitude_rad * np.array([1.0 if number % 2 == 0 else -1.0 for number in range(len(units))])
    arrays, gain = level_sequence(start, index, space.room(start), offsets, [unit * unit_time_s for unit in units], space.rate_hz,
                                  edge_fraction=edge_fraction, margin=margin, label=label)
    note = f"{label}: pulses of {pattern} x {unit_time_s:g} s at {deg(gain * amplitude_rad)}"
    if gain < 1.0:
        note += f" (asked {deg(amplitude_rad)}: lowered so every edge fits the velocity and acceleration limits and the room)"
    return Motion([arrays], [note])
