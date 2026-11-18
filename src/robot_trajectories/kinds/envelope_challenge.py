from __future__ import annotations

from ..registry import kind
from ..segments import annotate, chain
from .coverage_greedy_swings import coverage_greedy_swings
from .hold import hold
from .home import home
from .linear_chirp import linear_chirp
from .random_waypoints import random_waypoints
from .speed_ladder import speed_ladder
from .velocity_sweep import velocity_sweep


@kind(use_cases=("tracking_eval", "state_coverage", "speed_limits", "gravity_id", "smoke_test"), family="standard_test",
      starts_as="time_signal", dimensions=("1D",), unchecked=("collision_model",), higher_order=True)
def envelope_challenge(space, joint=None, *, sweep_speeds=(0.1, 0.5), sweep_span=0.95, ladder_rungs=(0.2, 0.4, 0.6, 0.8, 1.0),
                       ladder_repeats=2, ladder_held_out=1, random_moves=10, random_speed=0.8, seed=0, coverage_pct=60.0,
                       coverage_max_s=90.0, chirp_hz=(0.5, 2.0), chirp_s=15.0, settle_s=1.0, home_between=False, start=None):
    """How does one joint track across its whole envelope? Position, speed and acceleration each pushed to the limits the
    space states, in one session that starts and ends at home:

    1. slow constant-speed passes across ``sweep_span`` of the room (``sweep_speeds``): the gravity and friction torque at
       every angle, the quasi-static floor;
    2. a speed ladder A <-> B across the room at ``ladder_rungs`` of top speed: tracking and stops as speed rises to the limit;
    3. ``random_moves`` seeded random moves at ``random_speed``: ordinary motion anywhere in the room;
    4. coverage-greedy swings until ``coverage_pct`` of the (q, qd, qdd) cells are predicted visited (or ``coverage_max_s``);
    5. a constant-speed linear chirp over ``chirp_hz`` around home: the frequency response;
    6. home.

    ``home_between``: end every part at home (a minimum-jerk move back after the sweep, the ladder, the moves and the swings), so a
    player that releases the joint between rest-to-rest pieces leaves it where it rests by itself.

    A higher-order kind: six lower kinds chained, recorded as this one call, so the call (its keyword arguments) is the
    configuration and the recipe rebuilds the session sample for sample.

    WHY no robot words here (Zach, 2026-11-18): the kind is task-agnostic; a pendulum enters only as its limits. Opening the
    room to a full turn is the caller's ``JointSpace``, never a constant in this file."""
    index = space.joint_index(joint)
    name = space.joint_names[index]
    sweep = annotate(velocity_sweep(space, index, speeds_fraction=tuple(sweep_speeds), span_fraction=sweep_span, start=start),
                     f"slow passes across {100 * sweep_span:.0f} % of the room at {', '.join(f'{100 * s:.0f} %' for s in sweep_speeds)}"
                     " of top speed: gravity and friction torque at every angle")
    # WHY home_between (2026-11-18, the first full-turn run on CAN 11): the rack runner STOPs the board between plan steps (its post-stop
    # current read), so a pendulum left away from the bottom fell limp and swung +-75 deg; the next step's prelude caught it at 12.9 N.m and
    # 6.97 rad/s, 93 % of the overspeed guard. Ending each part at home leaves nothing to fall.
    def back(part):
        return chain(part, home(space, start=part)) if home_between else part

    sweep = back(sweep)
    settle_1 = hold(space, settle_s, start=sweep)
    ladder = back(annotate(speed_ladder(space, index, rung_fractions=tuple(ladder_rungs), repeats=ladder_repeats, held_out=ladder_held_out,
                                        start=settle_1),
                           f"A <-> B across the room at {', '.join(f'{100 * r:.0f} %' for r in ladder_rungs)} of top speed: "
                           "tracking and stops as speed rises"))
    settle_2 = hold(space, settle_s, start=ladder)
    moves = back(annotate(random_waypoints(space, random_moves, seed=seed, speed=random_speed, start=settle_2),
                          f"{random_moves} seeded random moves (seed {seed}) at {100 * random_speed:.0f} %: ordinary motion anywhere"))
    settle_3 = hold(space, settle_s, start=moves)
    swings = annotate(coverage_greedy_swings(space, index, target_coverage_pct=coverage_pct, max_duration_s=coverage_max_s,
                                             start=settle_3),
                      f"swings picked to fill the (q, qd, qdd) cells, aiming at {coverage_pct:.0f} %")
    to_home = home(space, start=swings)
    sweep_hz = annotate(linear_chirp(space, index, f_start_hz=chirp_hz[0], f_end_hz=chirp_hz[1], duration_s=chirp_s,
                                     start=to_home),
                        f"frequency response around home, {chirp_hz[0]:g}-{chirp_hz[1]:g} Hz at constant speed")
    back = annotate(home(space, start=sweep_hz), f"back home on {name}, so the next test begins where this one did",
                    wait_after=True)
    return chain(sweep, settle_1, ladder, settle_2, moves, settle_3, swings, to_home, sweep_hz, back)
