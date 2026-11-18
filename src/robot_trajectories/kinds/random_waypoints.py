from __future__ import annotations

from .. import kernels as k
from ..registry import kind


@kind(use_cases=("tracking_eval", "smoke_test"), unchecked=("collision_model",), catalog_id="random_targets")
def random_waypoints(space, count=8, *, seed=0, margin=0.1, speed=0.6, dwell_s=0.2, label="random waypoints", start=None):
    """Seeded random minimum-jerk moves inside the limits: everyday tracking.

    ``count`` configurations drawn uniformly inside the room around ``start`` shrunk by ``margin``; ``seed`` is part of the
    recipe, so the same call is the same motion. With a collision model in the space, the guard refuses a draw that collides.

    WHY the room: with a reach map the draw stays inside the reachable box around the start, so no leg crosses a hole."""
    if count < 1:
        raise ValueError(f"random_waypoints.count must be at least 1, got {count}")
    targets = k.random_waypoints(space.room(start), count, seed=seed, margin=margin)
    return k.waypoint_chain(targets, space.bounds, space.rate_hz, start=start, speed=speed, dwell_s=dwell_s, label=label)
