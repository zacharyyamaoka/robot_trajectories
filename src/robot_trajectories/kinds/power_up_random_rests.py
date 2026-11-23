from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..registry import LIMITS, kind
from ..segments import Motion
from ..time_parameterization import TimeParameterization
from . import _lane_b as b

STRATA = (("singular", 0.35), ("flipped", 0.25), ("free", 0.40))  # WHY oversampled: where a hand or a fall can leave an arm is not uniform
POOL = 8


def _conditioning(space, q: np.ndarray) -> float:
    """sigma_min / sigma_max of the tool position's Jacobian at ``q`` (central differences of ``space.forward``): 0 at a singularity."""
    columns = []
    for joint in range(space.dof):
        step = np.zeros(space.dof)
        step[joint] = 1e-5
        columns.append((space.forward(q + step)[:3, 3] - space.forward(q - step)[:3, 3]) / 2e-5)
    values = np.linalg.svd(np.array(columns).T, compute_uv=False)
    return float(values[-1] / values[0]) if values[0] > 0 else 0.0


@kind(use_cases=("homing", "smoke_test"), needs=(*LIMITS, "forward_kinematics"), unchecked=("collision_model",),
      catalog_id="power_up_random_rests")
def power_up_random_rests(space, *, power_ups=320, seed=20261110, strata=("singular", "flipped", "free"), dwell_s=0.5,
                          time_parameterization=TimeParameterization.minimum_jerk(peak_speed_fraction=0.5), label="random rests",
                          start=None):
    """A pose set of ``power_ups`` seeded rests, each a station to dwell at: configurations where a hand or a fall can leave the
    arm, with the awkward ones oversampled (singular 35 %, flipped 25 %, free 40 %), for a power-up or recovery path to be run
    from each.

    The strata are defined here without naming any robot: ``singular`` = the lowest sigma_min / sigma_max of the tool position's
    Jacobian in a pool of draws (the catalog's near-straight and near-fold cells); ``flipped`` = the draws farthest from the start
    in joint space (the other side of the room: the catalog's flipped knee and mirrored crank); ``free`` = uniform. A draw
    is admitted when the kinematics has a tool pose there and nothing collides. The stations and each one's stratum are in the
    segment entry. Seeded.

    WHY the stations, not the recovery: the generated part is the set of start states (sim only by design, an arm released
    unpowered falls into its rest); the legs between them only visit them."""
    unknown = [name for name in strata if name not in dict(STRATA)]
    if unknown or power_ups < 1 or dwell_s < 0:
        raise ValueError(f"power_up_random_rests needs power_ups >= 1, dwell_s >= 0 and strata among {[n for n, _ in STRATA]}; got "
                         f"{power_ups}, {dwell_s}, {strata}")
    shares = np.array([dict(STRATA)[name] for name in strata])
    counts = np.floor(shares / shares.sum() * power_ups).astype(int)
    counts[-1] += power_ups - int(counts.sum())
    rng = np.random.default_rng(seed)
    box = space.room(start).shrunk(0.05)
    span = np.maximum(space.bounds.high - space.bounds.low, 1e-9)

    def draw(count: int) -> list[np.ndarray]:
        found: list[np.ndarray] = []
        for _ in range(500 * max(count, 1)):
            q = rng.uniform(box.low, box.high)
            if b.tool_ok(space, q):
                found.append(q)
                if len(found) >= count:
                    return found
        raise ValueError(f"only {len(found)} of {count} admitted rests found in the room around the start")

    rests: list[tuple[np.ndarray, str]] = []
    for name, count in zip(strata, counts, strict=True):
        if count == 0:
            continue
        if name == "free":
            rests += [(q, name) for q in draw(int(count))]
            continue
        pool = draw(int(count) * POOL)
        if name == "singular":
            order = np.argsort([_conditioning(space, q) for q in pool])
        else:
            order = np.argsort([-float(np.linalg.norm((q - start) / span)) for q in pool])
        rests += [(pool[i], name) for i in order[: int(count)]]
    # nearest-neighbour visiting order (kernels.nearest_order's rule), keeping each rest's stratum
    remaining, here, ordered = list(range(len(rests))), np.asarray(start, float), []
    while remaining:
        pick = remaining.pop(int(np.argmin([float(np.max(np.abs(rests[i][0] - here))) for i in remaining])))
        ordered.append(rests[pick])
        here = rests[pick][0]
    stations = [q for q, _ in ordered]
    pieces = b.station_pieces(space, start, stations, time_parameterization, label=label,
                              dwell=lambda index, q: k.hold(q, dwell_s, space.rate_hz, label=f"{label}: station {index}"))
    entry = {**b.stations_entry(stations), "strata": [name for _, name in ordered]}
    return Motion(pieces, [f"{power_ups} rests: " + ", ".join(f"{int(c)} {n}" for n, c in zip(strata, counts, strict=True))], entry)
