from __future__ import annotations


from ..registry import LIMITS, kind
from ..segments import annotate, chain, repeat
from ..time_parameterization import TimeParameterization
from .hold import hold
from .joint_waypoints import joint_waypoints

DUTY_TYPES = ("S1", "S3")


@kind(use_cases=("endurance", "thermal", "repeatability"), needs=(*LIMITS, "home_pose"), catalog_id="endurance_duty_cycle", higher_order=True)
def endurance_duty_cycle(space, *, cycles=10, duty_type="S1", duty_fraction=0.6, reach_fraction=0.5, speed_fraction=0.5, dwell_s=0.2,
                         gap_check_every_s=43200.0, label="duty cycle", start=None):
    """One duty cycle played ``cycles`` times with a labelled gap check every ``gap_check_every_s`` of motion: wear, drift and
    thermal steady state, for hours of play.

    The cycle moves every joint out to ``reach_fraction`` of the room on one side, across to the other, and back to the start
    (rest-to-rest legs at ``speed_fraction`` of the limits, ``dwell_s`` at each stop). ``duty_type`` is IEC 60034's: ``S1`` is
    continuous running (no rest), ``S3`` is intermittent periodic duty, the cycle followed by a rest so that motion is
    ``duty_fraction`` of the period; the others (S2, S4-S10) are refused by name. The plays are one segment (``repeat``); each gap
    check is its own segment, a short hold marked ``wait_after`` (split and run, then check).

    WHY ``cycles`` defaults to 10, not the catalog's 10000: 10000 cycles is ~30 million samples a call; a player loops a cycle
    this long (``repeat``, or call this per block of hours). The count is the argument."""
    if duty_type not in DUTY_TYPES:
        raise ValueError(f"endurance_duty_cycle: duty_type {duty_type!r} is not built; IEC 60034's S1 and S3 are ({DUTY_TYPES})")
    if cycles < 1 or not 0 < duty_fraction <= 1 or not 0 < reach_fraction <= 1 or gap_check_every_s <= 0:
        raise ValueError(f"endurance_duty_cycle needs cycles >= 1, 0 < duty_fraction <= 1, 0 < reach_fraction <= 1, gap_check_every_s > 0; "
                         f"got {cycles}, {duty_fraction}, {reach_fraction}, {gap_check_every_s}")
    room = space.room(start)
    out = start + reach_fraction * (room.high - start)
    across = start - reach_fraction * (start - room.low)
    motion = annotate(joint_waypoints(space, [out, across],
                                      time_parameterization=TimeParameterization.minimum_jerk(peak_speed_fraction=speed_fraction),
                                      dwell_s=dwell_s, back_to_start=True, label=f"{label} motion", start=start),
                      f"the duty cycle's motion: out to {reach_fraction:g} of the room, across, back; {duty_type}")
    cycle = motion
    if duty_type == "S3" and duty_fraction < 1.0:
        moving_s = float(motion.t[-1])
        rest = annotate(hold(space, moving_s * (1.0 - duty_fraction) / duty_fraction, label=f"{label} rest", start=motion),
                        f"S3 rest: motion is {duty_fraction:g} of the period")
        cycle = chain(motion, rest)
    per_check = max(1, int(round(gap_check_every_s / float(cycle.t[-1]))))
    plays, left = [], cycles
    while left > 0:
        count = min(per_check, left)
        plays.append(repeat(cycle, count, label=f"{label}: {count} plays"))
        left -= count
        if left > 0:
            check = hold(space, 1.0, label=f"{label}: gap check", start=plays[-1])
            plays.append(annotate(check, "gap check: stop here and measure the backlash and drift, then resume", wait_after=True))
    return chain(*plays) if len(plays) > 1 else plays[0]
