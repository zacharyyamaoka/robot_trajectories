"""The kind contract as one call a kind's test makes: ``check_kind(trajectory, space)``. Every lane's per-kind test uses it.

    from kind_contract import check_kind
    check_kind(my_kind(space, joint="can11"), space)

It holds what docs/peps/0002-signal-library.md promises of every kind, independently of the decorator that is meant to
enforce it (an independent ruler: it re-derives each check from the samples, never trusts the guard):

1. a robot_msgs JointTrajectory with velocities and accelerations, on the space's joints, carrying its limits;
2. inside the position, velocity and acceleration limits, no jump between samples, at rest at both ends unless its kind
   declares a moving or event-ended end;
3. no collision when the space carries a checker;
4. its recipe rebuilds it sample for sample, in this space; and refuses another;
5. ``split`` then ``chain`` gives it back, and each split piece rebuilds alone from its own call;
6. no task words in its labels (Zach, "Robot Trajectories": a kind never knows what the motion is for).
"""

from __future__ import annotations

import json

import numpy as np
from robot_msgs.trajectory import JointTrajectory

from robot_trajectories import JointSpace, chain, rebuild, recipe, segment_info, split
from robot_trajectories.segments import PIECE

TASK_WORDS = ("grasp", "gripper", "belt", "conveyor", "item", "object", "bin", "tray", "part ")


def check_kind(trajectory: JointTrajectory, space: JointSpace, *, other_space: JointSpace | None = None) -> None:
    """Assert the kind contract on one output (see the module docstring). ``other_space``: a space it must be refused in."""
    assert isinstance(trajectory, JointTrajectory), type(trajectory)
    assert trajectory.velocities is not None and trajectory.accelerations is not None, "no velocities or accelerations"
    assert tuple(trajectory.joint_names) == space.joint_names, (trajectory.joint_names, space.joint_names)
    assert trajectory.limits is not None, "the trajectory does not carry its limits"
    limits = space.limits
    q, qd, qdd, t = trajectory.positions, trajectory.velocities, trajectory.accelerations, trajectory.t
    assert np.all(np.diff(t) > 0), "time does not increase"
    assert not np.any(q < limits.min_position - 1e-9) and not np.any(q > limits.max_position + 1e-9), "leaves its position limits"
    assert np.max(np.abs(qd) / limits.max_velocity) <= 1 + 1e-6, f"velocity {np.max(np.abs(qd) / limits.max_velocity):.4f}x the limit"
    assert np.max(np.abs(qdd) / limits.max_acceleration) <= 1 + 1e-6, f"acceleration {np.max(np.abs(qdd) / limits.max_acceleration):.4f}x"
    dt = np.diff(t)[:, None]
    assert not np.any(np.abs(np.diff(q, axis=0)) > limits.max_velocity * dt * 1.05 + 1e-9), "jumps in position"
    assert not np.any(np.abs(np.diff(qd, axis=0)) > limits.max_acceleration * dt * 1.05 + 1e-9), "jumps in velocity"
    entries = segment_info(trajectory)
    if entries[0].get("starts", "at_rest") == "at_rest":
        assert np.max(np.abs(qd[0])) <= 1e-9, "does not start at rest"
    if entries[-1].get("ends", "at_rest") == "at_rest":
        assert np.max(np.abs(qd[-1])) <= 1e-9, "does not end at rest"
    if space.checks_collision:
        hits = [row for row in range(0, len(t), 5) if space.in_collision(q[row])]
        assert not hits, f"collides at samples {hits[:5]}"
    again = rebuild(trajectory, space)
    assert np.array_equal(again.positions, q) and np.array_equal(again.t, t), "its recipe does not rebuild it"
    assert len(json.loads(recipe(trajectory))["calls"]) >= 1
    if other_space is not None:
        try:
            rebuild(trajectory, other_space)
        except ValueError:
            pass
        else:
            raise AssertionError("its recipe rebuilt in another space")
    pieces = split(trajectory)
    assert len(pieces) == len(entries)
    joined = chain(*pieces)
    assert np.array_equal(joined.positions, q) and np.array_equal(joined.velocities, qd), "chain(*split(x)) is not x"
    for piece, entry in zip(pieces, entries, strict=True):
        if entry.get("call") is not None:
            assert np.array_equal(rebuild(piece, space).positions, piece.positions), f"piece {entry['label']!r} does not rebuild alone"
    labels = [entry["label"] for entry in entries] + [piece["label"] for piece in json.loads(trajectory.header.meta["pieces"])]
    for label in labels:
        for word in TASK_WORDS:
            assert word not in f"{label.lower()} ", f"task word {word.strip()!r} in the label {label!r}"
    assert len(trajectory.channels[PIECE]) == len(t)
