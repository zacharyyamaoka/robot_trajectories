from __future__ import annotations

from ..registry import kind
from ..segments import annotate, chain
from .home import home
from .random_waypoints import random_waypoints


@kind(use_cases=("tracking_eval", "smoke_test"), family="coverage", starts_as="path", dimensions=("1D", "2D", "3D", "6D"),
      unchecked=("collision_model",), higher_order=True)
def tracking_check(space, *, count=8, seed=0, start=None):
    """Does it follow ordinary moves? ``count`` seeded random moves (tracking error, RMS and peak), then home, so the next test
    begins where this one did."""
    moves = annotate(random_waypoints(space, count, seed=seed, start=start),
                     f"{count} seeded random moves (seed {seed}): tracking error, RMS and peak")
    back = annotate(home(space, start=moves), "back home, so the next test begins where this one did", wait_after=True)
    return chain(moves, back)
