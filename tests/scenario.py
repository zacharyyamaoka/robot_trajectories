"""The round's shared scenario (2026-11-18), written in the signal-library API: a JointSpace in, kinds (plain functions) out.

    cd ~/robot_trajectories && /home/bam/robot_trajectories/.venv/bin/python tests/scenario.py

The motion is v6's, bit for bit (tests/test_equivalence.py); ``tests/test_scenario.py`` holds it to the round's ruler. What
changed from 0.0.1: plans and records became functions. ``pid_tuning(space, ...)`` is a higher-order kind (step train, hold,
chirp, hold, chained); ``chain`` joins; ``split`` / ``labels`` read the segments; the recipe text rebuilds it.
"""

from __future__ import annotations

import math

import numpy as np

from fixtures import robots  # the three robots, each in the form Zach named: bare limits, a reach map, a description
from robot_trajectories import (JointSpace, MissingRung, catalog, chain, dynamics_excitation, friction_id, labels, pid_tuning,
                                rebuild as rebuild_recipe, recipe, split, summary, tracking_check, workspace_survey)

# 1. What serves PID tuning? The kinds that generate it (and the catalog rows nothing generates yet, listed apart).
PID_KINDS = catalog(use_case="pid_tuning").implemented


# 2. The pendulum (CAN 11) from bare joint limits: three higher-order kinds chained.
def pendulum_space() -> JointSpace:
    return JointSpace.from_limits(robots.PENDULUM_LIMITS)


def pendulum():
    space = pendulum_space()
    return chain(pid_tuning(space, joint="can11", steps_fraction=(0.5, -0.5, 0.25, -0.25, 0.9), dwell_s=1.5),  # the round's steps
                 friction_id(space, joint="can11"),
                 tracking_check(space, count=8, seed=1))


# 3. The five-bar: joint limits plus its reach map, the optional cache of the kinematics rung.
def fivebar_space() -> JointSpace:
    return JointSpace.from_limits(robots.FIVEBAR_LIMITS, home=np.radians([100.0, -76.0])).with_reach_map(robots.fivebar_reach_map())


def fivebar():
    space = fivebar_space()
    return chain(workspace_survey(space), dynamics_excitation(space))  # every kind stays inside the hole-free room


# 4. The UR5e from its description. The URDF has no acceleration: refused until one is stated (and then recorded).
def ur5e_space() -> JointSpace:
    return JointSpace.from_description(robots.UR5E_DESCRIPTION, max_acceleration=6.0).within(robots.UR5E_DEMO_BOX)


def ur5e():
    return dynamics_excitation(ur5e_space(), band_hz=(0.1, 1.0), period_s=20.0, periods=1, pose_grid_joints=robots.UR5E_JOINTS[:3])


# 5. The recipe as text, and the session made again from that text alone (a space of joint limits is written out whole).
def rebuild(config_text: str):
    return rebuild_recipe(config_text)


def run() -> dict:
    session = pendulum()
    return {
        "catalog_pid": PID_KINDS.names,
        "pendulum": {"joined": session, "segments": split(session), "labels": labels(session)},
        "fivebar": {"joined": fivebar()},
        "ur5e": {"joined": ur5e()},
        "config": recipe(session),
    }


if __name__ == "__main__":
    print("1. catalog(use_case='pid_tuning'):")
    print(catalog(use_case="pid_tuning", space=pendulum_space()))
    session = pendulum()
    print("\n2. pendulum:", pendulum_space())
    print("   " + summary(session).replace("\n", "\n   "))
    five = fivebar()
    print("\n3. five-bar:", fivebar_space())
    print("   " + summary(five).replace("\n", "\n   "))
    try:
        JointSpace.from_description(robots.UR5E_DESCRIPTION)
    except MissingRung as refusal:
        print(f"\n4. UR5e without an acceleration: {refusal}")
    arm = ur5e()
    print("   " + summary(arm).replace("\n", "\n   "))
    text = recipe(session)
    again = rebuild(text)
    same = np.array_equal(again.t, session.t) and np.array_equal(again.positions, session.positions)
    print(f"\n5. recipe: {len(text)} characters of JSON; rebuilt from the text alone: {'identical' if same else 'DIFFERENT'}")
    try:
        rebuild_recipe(text, fivebar_space())
    except ValueError as refusal:
        print(f"   on the five-bar instead: {refusal}")
    print(f"   swing of the shared chirp: {math.degrees(0.5 * np.ptp(split(session)[2].positions)):.2f}° all the way from 0.2 to 5 Hz")
