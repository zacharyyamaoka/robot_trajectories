from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..registry import LIMITS, kind
from ..segments import Motion
from ._lane_a import quintic


@kind(use_cases=("tracking_eval", "smoke_test", "conveyor_pick_and_place"), needs=(*LIMITS, "home_pose"), catalog_id="rapid_replan")
def rapid_replan(space, *, replan_period_s=0.15, replans=8, target_spread_rad=0.349, move_s=0.6, seed=0, label="rapid replan", start=None):
    """A new plan every ``replan_period_s``, each starting from the previous plan's position, velocity and acceleration at that
    instant and ending at rest at a seeded target within ``target_spread_rad`` of the named home (the controller test: take a
    new trajectory mid-motion without a jump). It starts at rest at the start and ends at rest at the last target.

    Every plan is a quintic over ``move_s`` (lengthened by a tenth until it fits the limits, with a note). Plan k runs for one
    period and plan k + 1 takes over from its exact state, so every join but the first and last is mid-motion (the one join kind no
    other kind exercises); the last plan runs to rest.

    WHY one kind, not a chain of ``ends="moving"`` kinds: the hand-over state is the quintic's own value at the period, which
    this body already holds, so the pieces join exactly (position, speed and acceleration) with no leg between them
    (robot_controllers' rung R5.RapidReplan ``_replan_chain``)."""
    if replan_period_s <= 0 or replans < 1 or move_s < replan_period_s or target_spread_rad < 0:
        raise ValueError("rapid_replan needs replan_period_s > 0, replans >= 1, move_s >= replan_period_s and target_spread_rad >= 0")
    rate = space.rate_hz
    period_steps = max(2, int(round(replan_period_s * rate)))
    home = np.asarray(space.home, float)
    room = space.room(home)
    # WHY 0.9 of the room about home: a quintic that leaves rest overshoots its target slightly when it starts moving the
    # other way; the margin keeps the overshoot inside the limits and, with a reach map, inside the reachable box.
    low, high = home - 0.9 * (home - room.low), home + 0.9 * (room.high - home)
    rng = np.random.default_rng(seed)
    targets = np.clip(home + rng.uniform(-target_spread_rad, target_spread_rad, size=(int(replans), len(home))), low, high)
    bounds = space.bounds
    position, velocity, acceleration = np.asarray(start, float), np.zeros(len(home)), np.zeros(len(home))
    pieces, notes = [], []
    for number, target in enumerate(targets):
        last = number == len(targets) - 1
        steps = int(round(move_s * rate))
        for _ in range(40):
            total = steps / rate
            used = steps if last else period_steps
            t = np.arange(used + 1) / rate
            p, v, a = quintic(position, velocity, acceleration, target, np.zeros(len(home)), np.zeros(len(home)), total, t)
            fits = (np.all(p >= bounds.low - 1e-9) and np.all(p <= bounds.high + 1e-9) and np.all(np.abs(v) <= 0.95 * bounds.max_velocity)
                    and np.all(np.abs(a) <= 0.95 * bounds.max_acceleration))
            if fits:
                break
            steps = int(np.ceil(steps * 1.1))
        else:
            raise ValueError(f"rapid_replan: plan {number + 1} cannot reach its target inside the limits from its hand-over state")
        if steps != int(round(move_s * rate)):
            notes.append(f"{label}: plan {number + 1} lengthened to {steps / rate:.2f} s (asked {move_s:g} s) to fit the limits")
        pieces.append(k.Arrays(t, p, v, a, f"{label}: plan {number + 1}"))
        position, velocity, acceleration = p[-1], v[-1], a[-1]
    return Motion(pieces, notes, entry={"targets": [[float(value) for value in row] for row in targets],
                                        "replan_period_s": period_steps / rate})
