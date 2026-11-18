from __future__ import annotations


from .. import kernels as k
from ..registry import kind
from ._lane_a import joint_room


def staircase_levels(steps: int) -> list[int]:
    """The level indices visited after 0: up to +n, down to -n, back up to 0 (every inner level twice, once from each side)
    (rack ``backlash.staircase_levels``)."""
    return list(range(1, steps + 1)) + list(range(steps - 1, -steps - 1, -1)) + list(range(-steps + 1, 1))


@kind(use_cases=("backlash_id", "stop_accuracy", "friction_id", "repeatability"), catalog_id="staircase")
def staircase(space, joint=None, *, levels=7, range_fraction=0.85, move_speed_fraction=0.1, hold_s=1.5, cycles=1,
              label="staircase", start=None):
    """Climb to +A in ``levels`` levels (odd; 7 is -3..+3), descend to -A, climb back to the start, holding each level, so every
    inner level is reached once from below and once from above (ISO 230-2's bidirectional positioning; the rack's backlash
    staircase). ``cycles`` repeats the whole climb-descend-climb.

    ``range_fraction``: A as a fraction of the room on the smaller side. ``move_speed_fraction``: each move takes this fraction of
    the velocity and acceleration limits.

    WHY holds and not a continuous triangle (the rack's 2026-10-23 finding): a camera lags the encoder by ~32 ms, which at any useful
    speed is a hysteresis as wide as a gearbox's backlash; at a hold the output has settled on the flank the rotor last pushed
    it against, so the up-going and down-going holds at one level differ by the dead band, free of latency."""
    if levels < 3 or levels % 2 == 0 or cycles < 1 or hold_s <= 0 or not 0 < range_fraction <= 1 or move_speed_fraction <= 0:
        raise ValueError("staircase needs odd levels >= 3, cycles >= 1, hold_s > 0, range_fraction in (0, 1] and move_speed_fraction > 0")
    index = space.joint_index(joint)
    bounds = space.room(start)
    steps = (levels - 1) // 2
    size = range_fraction * joint_room(start, bounds, index) / steps
    pieces, here, level = [], k.hold(start, 0.0, space.rate_hz).q[0], 0
    for target_level in staircase_levels(steps) * int(cycles):
        target = here.copy()
        target[index] = start[index] + target_level * size
        side = "from below" if target_level > level else "from above"
        pieces.append(k.min_jerk(here, target, bounds, space.rate_hz, speed=move_speed_fraction,
                                 label=f"{label}: to level {target_level:+d} {side}"))
        pieces.append(k.hold(target, hold_s, space.rate_hz, label=f"{label}: hold level {target_level:+d} {side}"))
        here, level = target, target_level
    return pieces
