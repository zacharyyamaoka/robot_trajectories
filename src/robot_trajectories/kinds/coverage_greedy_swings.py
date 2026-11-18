from __future__ import annotations

import math

import numpy as np

from .. import kernels as k
from ..registry import kind
from ..segments import Motion
from ._lane_c import faded_sine

# bam_fivebar coverage_set.py's constants and sim2real.schedule.joint's cell edges (QD_EDGE_FRACTIONS, QDD_EDGE_FRACTIONS, Q_BINS)
Q_BINS = 12
QD_EDGE_FRACTIONS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
QDD_EDGE_FRACTIONS = (0.00625, 0.025, 0.0625, 0.1875, 0.5, 1.0)
SPEED_FRACTIONS = (0.1, 0.2, 0.35, 0.5, 0.7, 0.9)  # the peak speed a swing aims at, as a fraction of the joint's top speed
ACCELERATION_FRACTIONS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9)
SAMPLE_HZ = 100.0  # WHY: sim2real samples a real run at its 100 Hz plan rate, so coverage is predicted from every (rate / 100)-th tick
SAFETY = 0.97  # a swing's real peaks (the ramps add to the sine's) must stay under this fraction of the limits


def _edges(low: float, high: float, vmax: float, amax: float, acceleration_cells: bool) -> list[np.ndarray]:
    signed = lambda fractions, scale: np.array([-f * scale for f in reversed(fractions)] + [f * scale for f in fractions])  # noqa: E731
    edges = [np.linspace(low, high, Q_BINS + 1), signed(QD_EDGE_FRACTIONS, vmax)]
    return edges + [signed(QDD_EDGE_FRACTIONS, amax)] if acceleration_cells else edges


def _centres(edge: np.ndarray) -> np.ndarray:
    return 0.5 * (edge[:-1] + edge[1:])


def _occupied(samples: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    """A boolean array over the cells: which any sample falls in (each bin [lo, hi), the last [lo, hi]; outside samples dropped)."""
    keep = np.ones(len(samples), bool)
    indices = []
    for axis, edge in enumerate(edges):
        index = np.searchsorted(edge, samples[:, axis], side="right") - 1
        index = np.where(samples[:, axis] == edge[-1], len(edge) - 2, index)
        keep &= (index >= 0) & (index < len(edge) - 1)
        indices.append(index)
    cells = np.zeros(tuple(len(edge) - 1 for edge in edges), bool)
    cells[tuple(index[keep] for index in indices)] = True
    return cells


@kind(use_cases=("state_coverage", "inertia_id", "friction_id"), unchecked=("collision_model",), catalog_id="coverage_greedy_swings")
def coverage_greedy_swings(space, joint=None, *, target_coverage_pct=40.0, acceleration_cells=True, max_swings=24, max_duration_s=240.0,
                           max_hz=2.0, margin_fraction=0.05, ramp_cycles=1.0, full_cycles=3.0, label="coverage swings", start=None):
    """Tapered sine swings on one joint, each at its own (centre, amplitude, frequency), picked greedily by how many empty cells of
    the joint's (q, qd, qdd) space each adds, until the predicted coverage clears ``target_coverage_pct`` (the five-bar crank's
    coverage set: bam_fivebar coverage_set.pick, sim2real.schedule.joint's cells). The other joints stay at ``start``.

    The cells: q in 12 bins over the joint's room, qd and qdd signed and log-spaced as fractions of its top speed and acceleration
    (a dead cell around 0); with ``acceleration_cells=False`` only (q, qd). The reachable cells are those whose speed the joint
    can brake from before the end of its room. A swing is centre + A w(t) sin(2 pi f t), w a raised-cosine ramp of ``ramp_cycles``
    in and out around ``full_cycles`` full ones (so it starts and ends at rest at its centre); its peak speed v and acceleration a
    set A = v^2 / a and f = a / (2 pi v), for a ladder of v and a as fractions of the limits, f at most ``max_hz``, A and
    ``margin_fraction`` of the room inside it. Coverage is predicted from the commanded samples at 100 Hz, as a run's are read.

    Stops at the target, at ``max_swings`` or ``max_duration_s`` of swings, or when no swing adds a cell; the notes give the
    coverage reached. Consecutive swings at different centres are joined by minimum-jerk transitions (not counted in the coverage)."""
    index = space.joint_index(joint)
    room = space.room(start)
    low, high = float(room.low[index]), float(room.high[index])
    vmax, amax = float(room.max_velocity[index]), float(room.max_acceleration[index])
    if high - low <= 0:
        raise ValueError(f"coverage_greedy_swings: {space.joint_names[index]} has no room around its start")
    edges = _edges(low, high, vmax, amax, acceleration_cells)
    grids = np.meshgrid(*(_centres(edge) for edge in edges), indexing="ij")
    qd_centre = grids[1]
    brake_room = np.where(qd_centre > 0, high - grids[0], grids[0] - low)
    reachable = qd_centre**2 / (2.0 * amax) <= brake_room + 1e-12
    stride = max(1, int(round(space.rate_hz / SAMPLE_HZ)))
    margin = margin_fraction * (high - low)
    candidates: list[tuple[k.Arrays, np.ndarray, tuple[float, float, float]]] = []
    for speed in SPEED_FRACTIONS:
        for acceleration in ACCELERATION_FRACTIONS:
            v, a = speed * vmax, acceleration * amax
            hz = a / v / (2.0 * math.pi)
            amplitude = v * v / a
            if hz > max_hz or 2.0 * (amplitude + margin) > high - low:
                continue
            first, last = low + margin + amplitude, high - margin - amplitude
            for centre in np.linspace(first, last, max(1, int(math.ceil((last - first) / max(amplitude, 0.05))) + 1)):
                base = np.array(start, float)
                base[index] = centre
                swing = faded_sine(base, index, amplitude, hz, ramp_cycles, full_cycles, space.rate_hz,
                                   label=f"{label}: {math.degrees(amplitude):.0f}° at {hz:.2f} Hz about {math.degrees(centre):+.0f}°")
                if np.max(np.abs(swing.qd[:, index])) > SAFETY * vmax or np.max(np.abs(swing.qdd[:, index])) > SAFETY * amax:
                    continue
                rows = [swing.q[::stride, index], swing.qd[::stride, index]] + ([swing.qdd[::stride, index]] if acceleration_cells else [])
                candidates.append((swing, _occupied(np.column_stack(rows), edges) & reachable, (v, a, amplitude)))
    if not candidates:
        raise ValueError(f"coverage_greedy_swings: no swing fits {space.joint_names[index]}'s room of "
                         f"{math.degrees(high - low):.1f}° under max_hz={max_hz}")
    have = np.zeros(reachable.shape, bool)
    chosen: list[int] = []
    seconds = 0.0
    total = int(reachable.sum())
    while len(chosen) < max_swings and 100.0 * have.sum() / total < target_coverage_pct:
        gains = [(-1 if i in chosen else int((cells & ~have).sum())) for i, (_, cells, _) in enumerate(candidates)]
        best = int(np.argmax(gains))
        if gains[best] <= 0 or seconds + candidates[best][0].t[-1] > max_duration_s:
            break
        chosen.append(best)
        seconds += float(candidates[best][0].t[-1])
        have |= candidates[best][1]
    if not chosen:
        raise ValueError("coverage_greedy_swings: no swing adds a cell inside max_duration_s")
    reached = 100.0 * have.sum() / total
    verdict = "clears" if reached >= target_coverage_pct else "falls short of"
    return Motion([candidates[i][0] for i in chosen],
                  [f"{label}: {len(chosen)} swings, {seconds:.0f} s, predicted {reached:.1f} % of {total} reachable "
                   f"({'q, qd, qdd' if acceleration_cells else 'q, qd'}) cells on {space.joint_names[index]}: {verdict} the "
                   f"{target_coverage_pct:g} % target"])
