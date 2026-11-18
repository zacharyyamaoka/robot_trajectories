"""The round's shared scenario (2026-11-18), written in this package's API: a workspace in, plans (or records) in, a Program out.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/scenario.py

Ported from bam_ws ``docs/trajectory_generator/variants/v6_splice/scenario.py`` (b5e1f6965). What changed: the imports (the
package instead of the round's sys.path), and the catalog rows come from the package's own ``trajectory_catalog``. The
motion is v6's, bit for bit (tests/test_equivalence.py); ``tests/test_scenario.py`` holds it to the round's ruler.
"""

from __future__ import annotations

import math

import numpy as np

from fixtures import robots  # the three robots, each in the form Zach named: bare limits, a reach map, a description
from robot_trajectories import JointTrajectoryGenerator, MissingInput, Program, Workspace, catalog, plans, trajectory_catalog

# 1. What serves PID tuning? The plans (what to run), and the record classes underneath them.
PID_PLANS = plans.for_use_case("pid_tuning")
PID_KINDS = catalog(use_case="pid_tuning")

# 2. The pendulum (CAN 11) from bare joint limits. The session is three purpose plans added up: configuration, no robot.
PENDULUM_SESSION = (plans.pid_tuning(joint="can11", steps_fraction=(0.5, -0.5, 0.25, -0.25, 0.9), dwell_s=1.5)  # the round's steps
                    + plans.friction_id(joint="can11")
                    + plans.tracking_check(count=8, seed=1))


def pendulum() -> Program:
    generator = JointTrajectoryGenerator(Workspace.from_limits(robots.PENDULUM_LIMITS))
    return generator.generate(PENDULUM_SESSION)  # .trajectory joined, .segments one per record, .steps purpose + wait


# 3. The five-bar: joint limits plus its reach map, the optional cache of the kinematics rung.
def fivebar_workspace() -> Workspace:
    return Workspace.from_limits(robots.FIVEBAR_LIMITS, home=np.radians([100.0, -76.0])).with_reach_map(robots.fivebar_reach_map())


def fivebar() -> Program:
    generator = JointTrajectoryGenerator(fivebar_workspace())
    return generator.generate(plans.workspace_survey() + plans.dynamics_excitation())  # every kind stays reachable


# 4. The UR5e from its description. The URDF has no acceleration: refused until one is supplied (and then logged).
def ur5e_workspace() -> Workspace:
    return Workspace.from_description(robots.UR5E_DESCRIPTION, max_acceleration=6.0).within(robots.UR5E_DEMO_BOX)


def ur5e() -> Program:
    generator = JointTrajectoryGenerator(ur5e_workspace())
    return generator.generate(plans.dynamics_excitation(band_hz=(0.1, 1.0), period_s=20.0, periods=1,
                                                        pose_grid_joints=robots.UR5E_JOINTS[:3]))


# 5. The configuration as text, and the session made again from that text alone.
def rebuild(config_text: str):
    return Program.from_json(config_text).trajectory


def run() -> dict:
    session = pendulum()
    return {
        "catalog_pid": PID_KINDS.ids,
        "pendulum": {"joined": session.trajectory, "segments": list(session.segments), "labels": session.labels},
        "fivebar": {"joined": fivebar().trajectory},
        "ur5e": {"joined": ur5e().trajectory},
        "config": session.to_json(),
    }


if __name__ == "__main__":
    catalog_rows = trajectory_catalog.rows()
    print("1. plans.for_use_case('pid_tuning'):", ", ".join(PID_PLANS), "| kinds:", ", ".join(PID_KINDS.ids))
    session = pendulum()
    print(session.generator.kinds("pid_tuning", catalog_rows=catalog_rows))
    print("\n2. pendulum:", session.generator.workspace.describe(catalog_rows))
    print("\n   " + PENDULUM_SESSION.describe().replace("\n", "\n   "))
    print("\n   " + session.summary().replace("\n", "\n   "))
    five = fivebar()
    print("\n3. five-bar:", five.generator.workspace)
    print("   " + five.summary().replace("\n", "\n   "))
    box = five.segments[2].positions
    print(f"   workspace sweep: {sum(b.label.startswith('workspace sweep: leg') for b in five.bands)} neighbour legs; multisine spans "
          f"{np.round(np.degrees(np.ptp(box, axis=0)), 1).tolist()}° inside the hole-free box")
    try:
        Workspace.from_description(robots.UR5E_DESCRIPTION)
    except MissingInput as refusal:
        print(f"\n4. UR5e without an acceleration: {refusal}")
    arm = ur5e()
    print("   " + str(arm.generator.workspace).replace("\n", "\n   "))
    print("   " + arm.summary().replace("\n", "\n   "))
    text = session.to_json()
    again = rebuild(text)
    same = np.array_equal(again.t, session.trajectory.t) and np.array_equal(again.positions, session.trajectory.positions)
    print(f"\n5. config: {len(text)} characters of JSON, {len(session.steps)} steps; rebuilt from the text alone: "
          f"{'identical' if same else 'DIFFERENT'}")
    try:
        Program.from_json(text, generator=five.generator)
    except ValueError as refusal:
        print(f"   on the five-bar instead: {refusal}")
    print(f"   swing of the shared chirp: {math.degrees(0.5 * np.ptp(session.segments[2].positions)):.2f}° all the way from 0.2 to 5 Hz")
