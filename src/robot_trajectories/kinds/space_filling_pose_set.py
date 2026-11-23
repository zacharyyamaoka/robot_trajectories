from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..registry import LIMITS, kind
from ..segments import Motion
from ..time_parameterization import TimeParameterization
from ._lane_c import SEQUENCES, unit_points

DRAW_FACTOR = 8  # WHY: with a collision checker, up to this many times the count are drawn, to replace the stations that collide


@kind(use_cases=("workspace_coverage", "kinematic_calibration", "gravity_id", "homing"), needs=LIMITS, unchecked=("collision_model",),
      catalog_id="space_filling_pose_set")
def space_filling_pose_set(space, count=64, *, method="sobol", seed=0, order="nearest", margin_fraction=0.05,
                           time_parameterization=TimeParameterization.minimum_jerk(peak_speed_fraction=0.5), dwell_s=0.5,
                           label="space-filling pose set", start=None):
    """``count`` stations spread evenly over the space's room (the limit box, or with a reach map the hole-free box around the
    start), visited one after another with a dwell at each: a pose set. The stations are a Sobol or Halton sequence or a Latin
    hypercube (``method``; numpy only), so ``count`` poses cover the room with no clusters; ``seed`` scrambles them (a random
    digital shift for Sobol, a random shift for Halton, the strata pairing for the hypercube). Sobol wants ``count`` a power of two.

    The stations are inside the room shrunk by ``margin_fraction`` of each joint's span. ``order``: "nearest" visits them by
    greedy nearest neighbour from the start (short legs), "sequence" in the order drawn. The legs are ``time_parameterization``'s rest-to-rest
    legs; each station's dwell is labelled "<label>: station k" and the stations' configurations are in the segment entry
    (``entry["stations"]``, rad).

    With a collision checker in the space a station that collides is replaced by the next draw, and a station whose leg would
    collide is left out; the notes say how many. WHY not a planner: a leg is a straight joint-space move, as in every pose set."""
    if count < 1:
        raise ValueError(f"space_filling_pose_set.count must be at least 1, got {count}")
    if order not in ("nearest", "sequence"):
        raise ValueError(f"space_filling_pose_set.order is 'nearest' or 'sequence', got {order!r}")
    if method not in SEQUENCES:
        raise ValueError(f"space_filling_pose_set.method is one of {SEQUENCES}, got {method!r}")
    box = space.room(start).shrunk(margin_fraction)
    draws = count * DRAW_FACTOR if space.checks_collision else count
    unit = unit_points(method, draws, space.dof, seed)
    configs = box.low + unit * (box.high - box.low)
    notes: list[str] = []
    if space.checks_collision:
        free = np.array([not space.in_collision(q) for q in configs])
        if not free.any():
            raise ValueError(f"space_filling_pose_set: all {draws} stations drawn collide (seed {seed})")
        configs = configs[free][:count]
        if len(configs) < count:
            notes.append(f"{label}: {len(configs)} of {count} stations are collision-free in {draws} draws")
    if order == "nearest":
        configs = k.nearest_order(configs, start)
    pieces: list[k.Arrays] = []
    stations: list[list[float]] = []
    here = start
    left_out = 0
    for q in configs:
        leg = time_parameterization.leg(here, q, space.limits, space.rate_hz, label=f"{label}: leg {len(stations) + 1}",
                                        dynamics=space.dynamics)
        if space.checks_collision and any(space.in_collision(row) for row in leg.q[::5]):
            left_out += 1
            continue
        pieces.append(leg)
        pieces.append(k.hold(q, dwell_s, space.rate_hz, label=f"{label}: station {len(stations) + 1}"))
        stations.append([float(v) for v in q])
        here = q
    if not stations:
        raise ValueError("space_filling_pose_set: every leg to a station collides")
    if left_out:
        notes.append(f"{label}: {left_out} stations left out, their leg collides")
    notes.append(f"{label}: {len(stations)} {method} stations, seed {seed}, visited in {order} order")
    return Motion(pieces, notes, entry={"stations": stations})
