"""envelope_challenge: one joint pushed across its whole envelope (position, speed, acceleration), home to home.

    taskset -c 2-5,16,17 nice -n 19 /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_envelope_challenge.py

The ruler is independent of the kind: the reach in position, speed and acceleration is read straight off the samples against
the space's own limits, and the (q, qd) coverage is a plain numpy histogram, not the kind's greedy predictor.
"""

from __future__ import annotations

import math

import numpy as np

import _runner
from kind_contract import TASK_WORDS, check_kind
from robot_msgs.trajectory import JointLimits
from robot_trajectories import JointSpace, kinds, rebuild, recipe, segment_info, split
from robot_trajectories.kinds.envelope_challenge import envelope_challenge


def full_turn(acceleration: float = 7.3896, speed: float = 5.0) -> JointSpace:
    """A one-joint space that may turn a full circle about home (the caller opens the room; the kind never does)."""
    limits = JointLimits(joint_names=("j",), min_position=(-math.pi,), max_position=(math.pi,), max_velocity=(speed,),
                         max_acceleration=(acceleration,))
    return JointSpace.from_limits(limits, home=(0.0,))


def narrow() -> JointSpace:
    limits = JointLimits(joint_names=("j",), min_position=(-0.33,), max_position=(0.33,), max_velocity=(2.0,),
                         max_acceleration=(7.3896,))
    return JointSpace.from_limits(limits)


def test_registered_as_higher_order_kind() -> None:
    info = kinds()["envelope_challenge"]
    assert info.higher_order, info
    assert not any(word in "envelope_challenge" for word in TASK_WORDS)


def test_contract_on_a_full_turn_and_a_narrow_window() -> None:
    for space in (full_turn(), narrow()):
        check_kind(envelope_challenge(space, "j"), space)


def test_reaches_the_envelope() -> None:
    space = full_turn()
    trajectory = envelope_challenge(space, "j")
    q, qd, qdd = trajectory.positions[:, 0], trajectory.velocities[:, 0], trajectory.accelerations[:, 0]
    assert q.min() <= -0.9 * math.pi and q.max() >= 0.9 * math.pi, np.degrees([q.min(), q.max()])
    assert np.abs(qd).max() >= 0.99 * 5.0, np.abs(qd).max()
    assert np.abs(qdd).max() >= 0.95 * 7.3896, np.abs(qdd).max()


def test_home_to_home_and_rebuilds() -> None:
    space = full_turn()
    trajectory = envelope_challenge(space, "j")
    assert abs(trajectory.positions[0, 0]) < 1e-12 and abs(trajectory.positions[-1, 0]) < 1e-12
    assert np.array_equal(rebuild(recipe(trajectory)).positions, trajectory.positions)
    assert len(segment_info(trajectory)) == 10 and len(split(trajectory)) >= 1


def test_session_covers_more_than_any_one_part() -> None:
    """(q, qd) cells, 24 x 20 over the limits: the whole session visits more than its best single segment."""
    space = full_turn()
    trajectory = envelope_challenge(space, "j")
    edges = (np.linspace(-math.pi, math.pi, 25), np.linspace(-5.0, 5.0, 21))

    def cells(rows: np.ndarray) -> set[tuple[int, int]]:
        q, qd = trajectory.positions[rows, 0], trajectory.velocities[rows, 0]
        histogram, _, _ = np.histogram2d(q, qd, bins=edges)
        return set(zip(*np.nonzero(histogram)))

    segment = np.asarray(trajectory.channels["segment"]).astype(int)
    whole = cells(np.arange(len(segment)))
    best = max(len(cells(np.flatnonzero(segment == s))) for s in np.unique(segment))
    assert len(whole) > best, (len(whole), best)


def test_seed_changes_only_the_random_moves() -> None:
    space = full_turn()
    a, b = envelope_challenge(space, "j", seed=0), envelope_challenge(space, "j", seed=1)
    assert np.array_equal(a.positions[:2000], b.positions[:2000])  # the sweep that opens it does not depend on the seed
    assert a.t[-1] != b.t[-1] or not np.array_equal(a.positions, b.positions)


def test_home_between_ends_every_part_at_home() -> None:
    """With home_between, the motion before every hold (and the session's end) is at rest at home: a player that releases the joint at a
    hold, or between the pieces a hold separates, leaves it where it rests by itself."""
    space = full_turn()
    trajectory = envelope_challenge(space, "j", home_between=True)
    check_kind(trajectory, space)
    entries = segment_info(trajectory)
    segment = np.asarray(trajectory.channels["segment"]).astype(int)
    for index, entry in enumerate(entries):
        following = entries[index + 1]["call"]["kind"] if index + 1 < len(entries) else None
        if entry["call"]["kind"] != "hold" and following in ("hold", None):
            last = np.flatnonzero(segment == index)[-1]
            assert abs(trajectory.positions[last, 0]) < 1e-9, (index, entry["call"]["kind"], trajectory.positions[last, 0])


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), __file__))
