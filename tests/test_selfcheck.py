"""v6's own claims, checked (beyond the round's ruler): 118 of them. Ported from bam_ws
docs/trajectory_generator/variants/v6_splice/selfcheck.py (b5e1f6965).

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_selfcheck.py

What changed: the scenario, the bare-kernel reference and catalog.json come from this repo (tests/scenario.py,
tests/fixtures/reference.py, robot_trajectories.trajectory_catalog) instead of the round's folders; the claims are the
round's lines, inside one function so importing the file runs nothing. As a script it prints one line per claim.
"""

from __future__ import annotations

import functools

import numpy as np

import scenario
from fixtures import reference, robots
from robot_trajectories import (DeploymentMotionClasses, Hold, JointTrajectoryGenerator, KindLocked, LinearChirp, MissingInput,
                                Move, Plan, Program, Repeat, SpeedLadder, TimeLaw, Workspace, WorkspaceSweep, catalog, plans,
                                trajectory_catalog)


@functools.lru_cache(maxsize=1)
def claims() -> tuple[tuple[bool, str], ...]:
    """Every claim, computed once: (holds, text)."""
    results: list[tuple[bool, str]] = []

    def claim(ok: bool, text: str) -> None:
        results.append((bool(ok), text))

    def inside(trajectory, limits) -> str:
        q, qd, qdd = trajectory.positions, trajectory.velocities, trajectory.accelerations
        problems = []
        if np.any(q < limits.min_position - 1e-9) or np.any(q > limits.max_position + 1e-9):
            problems.append("position")
        if np.any(np.abs(qd) > limits.max_velocity * (1 + 1e-6)):
            problems.append(f"velocity {np.max(np.abs(qd) / limits.max_velocity):.3f}x")
        if np.any(np.abs(qdd) > limits.max_acceleration * (1 + 1e-6)):
            problems.append(f"acceleration {np.max(np.abs(qdd) / limits.max_acceleration):.3f}x")
        dt = np.diff(trajectory.t)[:, None]
        if np.any(np.abs(np.diff(qd, axis=0)) > limits.max_acceleration * dt * 1.05 + 1e-9):
            problems.append("velocity jump")
        if np.max(np.abs(qd[[0, -1]])) > 1e-6:
            problems.append("not at rest at the ends")
        return ", ".join(problems)

    # 1. the ruler's motion is the reference's, sample for sample (pendulum) ------------------------------------------------
    ours = scenario.pendulum()
    theirs, _, _ = reference.pendulum_session(reference.PENDULUM_CONFIG)
    claim(np.array_equal(ours.trajectory.positions, theirs.positions) and np.array_equal(ours.trajectory.t, theirs.t),
          "pendulum session = the bare-kernel reference, every sample")
    claim(np.array_equal(scenario.ur5e().trajectory.positions, reference.ur5e_excitation().positions), "UR5e session = the reference, every sample")

    # 2. every kind, defaults, on every robot: inside the limits, at rest at both ends, five-bar reachable -------------------
    robots_under_test = {"pendulum": scenario.Workspace.from_limits(robots.PENDULUM_LIMITS), "five-bar": scenario.fivebar_workspace(),
                         "UR5e": scenario.ur5e_workspace()}
    joint_of = {"pendulum": "can11", "five-bar": "j2", "UR5e": "elbow_joint"}
    for robot, workspace in robots_under_test.items():
        generator = JointTrajectoryGenerator(workspace)
        for kind in catalog():
            fields = {spec.name for spec in kind.__dataclass_fields__.values()}
            options = {"joint": joint_of[robot]} if "joint" in fields else {}
            if kind is Repeat:
                options = {"records": (Move(joint=joint_of[robot], targets_fraction=(0.5,)),), "times": 3}
            if kind is SpeedLadder and robot == "five-bar":
                options["margin_rad"], options["margin_fraction"] = 0.0, 0.05  # its hole-free box is 6.3° wide: no 5° margins
            record = kind(**options)
            try:
                program = generator.generate(record)
            except (ValueError, KindLocked) as error:
                claim(False, f"{robot}: {kind.__name__} generates ({error})")
                continue
            problem = inside(program.trajectory, workspace.limits)
            if robot == "five-bar" and not problem:
                off = [q for q in program.trajectory.positions[::5] if (p := robots.fivebar_passive(*q)) is None or p[2] < 0.2]
                problem = f"leaves the reachable set at {len(off)} samples" if off else ""
            claim(not problem, f"{robot}: {kind.__name__} inside every limit, at rest at both ends{' (' + problem + ')' if problem else ''}")
            again = Program.from_json(program.to_json(), generator=generator)
            claim(np.array_equal(again.trajectory.positions, program.trajectory.positions), f"{robot}: {kind.__name__} JSON round trip")

    # 3. the speed ladder is one path under several time laws; the acceleration clamp is reported ---------------------------
    ladder = SpeedLadder(joint="can11")
    claim([law.peak_speed_fraction for law in ladder.time_laws] == [0.2, 0.4, 0.6, 0.8, 0.9, 1.0], "SpeedLadder.time_laws: one TimeLaw per rung")
    pendulum = JointTrajectoryGenerator(robots_under_test["pendulum"])
    program = pendulum.generate(ladder)
    legs = [band for band in program.bands if band.label.endswith("A to B")]
    claim(len(legs) == 6 * 7, f"ladder: 6 rungs x (5 repeats + 2 held out) A-to-B legs ({len(legs)})")
    repeat_legs = [band for band in legs if "repeat" in band.label and band.label.startswith("speed ladder: 20 %")]
    spans = {round(band.end_s - band.start_s, 6) for band in repeat_legs}
    claim(len(spans) == 1, f"ladder: the 20 % rung's 5 repeats are the same input (leg durations {sorted(spans)})")
    held = [band.label for band in legs if "held out" in band.label]
    claim(held[0].endswith("(seed 0): A to B") and "seed 51" in held[-1], f"ladder: held-out seeds are round x 1000 + rung x 10 + k ({held[0]!r} ... {held[-1]!r})")
    claim(any("acceleration-bound" in note for note in program.notes), "ladder: rungs above what the acceleration allows are noted")
    law = TimeLaw.minimum_jerk(peak_speed_fraction=1.0)
    timing = law.leg_timing(np.array([-0.24]), np.array([0.24]), pendulum.workspace.bounds)
    claim(timing.bound == "acceleration" and timing.peak_speed_fraction < 0.8, f"TimeLaw clamps to the acceleration limit ({timing})")
    try:
        TimeLaw("trapezoid", peak_speed_fraction=0.5)
        claim(False, "an unbuilt time law is refused")
    except ValueError as error:
        claim("not built yet" in str(error), "an unbuilt time law is refused by name")

    # 4. the new chirp: bigger at low frequency than the shared one, never beyond the acceleration ceiling --------------------
    week = pendulum.generate(LinearChirp(joint="can11", f_start_hz=0.2, f_end_hz=5.0, hold_after_s=0.0)).segments[0]
    shared = pendulum.generate(scenario.PENDULUM_SESSION[2].record).segments[0]
    claim(np.ptp(week.positions) > 20 * np.ptp(shared.positions), f"constant-speed chirp swings {np.ptp(week.positions) / np.ptp(shared.positions):.0f}x the shared chirp")
    claim(not inside(week, pendulum.workspace.limits), "constant-speed chirp inside every limit")

    # 5. repeat: every repeat is the same input, from the same place --------------------------------------------------------
    tape = Move(joint="can11", targets_fraction=(0.6,), back_to_start=False)  # ends away from its start
    program = pendulum.generate(Repeat(records=(tape,), times=3))
    backs = [band for band in program.bands if band.label.endswith("back to start")]
    claim(len(backs) == 2, f"Repeat moves back to the start between repeats when the tape ends elsewhere ({len(backs)} moves)")

    # 6. deployment classes: PID tuna's draws; held_out() is the sealed seeds -------------------------------------------------
    classes = DeploymentMotionClasses(joint="can11")
    claim(classes.held_out().seeds == (3, 4, 5) and classes.seeds == (0, 1, 2), "deployment classes: train seeds 0-2, held_out() 3-5")
    legs = classes.legs("pickplace", __import__("random").Random(0))
    claim(len(legs) == 5 and all(0.4 <= abs(target) <= 0.95 for target, _, _ in legs[:4]), "deployment classes: pickplace is 4 targets at 40-95 % of the room + home")

    # 7. plans: robot-free, add with +, round-trip as text, the last step of each waits ---------------------------------------
    session = scenario.PENDULUM_SESSION
    claim(Plan.from_json(session.to_json()) == session, "a plan round-trips through its JSON text")
    claim([item.then for item in session].count("wait") == 3, "each plan's last step says then wait (3 plans, 3 waits)")
    claim(session.plans == ["pid_tuning", "friction_id", "tracking_check"], "a sum of plans keeps whose step is whose")
    five = JointTrajectoryGenerator(scenario.fivebar_workspace())
    claim(len(five.generate(plans.pid_tuning(joint="j2")).segments) == 4, "the same plan runs on another robot (pid_tuning on the five-bar)")

    # 8. the workspace ladder: refusals name the fix; slots unlock catalog rows; a different robot is refused -----------------
    try:
        Workspace.from_description(robots.UR5E_DESCRIPTION)
        claim(False, "a description without acceleration is refused")
    except MissingInput as error:
        claim("max_acceleration=" in str(error), "a description without acceleration is refused, naming max_acceleration=")

    class FakeKinematics:  # bam_reach.Kinematics' shape, nothing more
        name, base_frame, cache_key = "fake fk", "base", "fake-1"
        joint_names = robots.UR5E_JOINTS

        def forward(self, joint_positions):
            return None

        def inverse(self, target_pose, seed=None):
            return None

    class FakeDynamics:  # pin_utils.PinDynamics' shape
        def inverse_dynamics(self, q, velocity, acceleration):
            return np.zeros_like(np.asarray(q, float))

    rows = trajectory_catalog.rows()
    arm = scenario.ur5e_workspace()
    before = sum(arm.row_status(row).state != "locked" for row in rows)
    after_k = sum(arm.with_kinematics(FakeKinematics()).row_status(row).state != "locked" for row in rows)
    after_d = sum(arm.with_kinematics(FakeKinematics()).with_dynamics(FakeDynamics()).row_status(row).state != "locked" for row in rows)
    claim(before < after_k < after_d, f"filling the slots unlocks catalog rows: {before} -> {after_k} (+ kinematics) -> {after_d} (+ dynamics) of {len(rows)}")
    claim(arm.with_kinematics(FakeKinematics()).kind_status(WorkspaceSweep).state == "fallback",
          "kinematics alone does not build a reachable set: workspace_sweep still falls back (nothing generates from the slot)")
    claim(robots_under_test["pendulum"].kind_status(Hold).state == "available", "Hold is available from bare joint limits")
    text = scenario.pendulum().to_json()
    try:
        Program.from_json(text, generator=JointTrajectoryGenerator(Workspace.from_limits(robots.PENDULUM_LIMITS, max_acceleration=5.0)))
        claim(False, "a program is refused on other limits")
    except ValueError as error:
        claim("another robot or limits" in str(error), "a program is refused on other limits (fingerprint)")
    logged = scenario.pendulum().trajectory
    claim(np.array_equal(Program.from_trajectory(logged).trajectory.positions, logged.positions), "Program.from_trajectory(logged) rebuilds it")

    # 9. adding a kind is one file: sine_dwell.py, found and listed with no other edit --------------------------------------
    claim("sine_dwell" in catalog(use_case="bandwidth").ids and catalog()["faded_sine"].__name__ == "SineDwell",
          "SineDwell (one new file) is registered, tagged bandwidth, mapped to catalog row faded_sine")
    claim(len({kind.catalog_id for kind in catalog() if kind.catalog_id}) == sum(1 for kind in catalog() if kind.catalog_id),
          "no catalog row is claimed by two kinds")
    claim(all(kind.catalog_id in {row["id"] for row in rows} for kind in catalog() if kind.catalog_id), "every claimed catalog row exists in catalog.json")
    now = {row["id"] for row in rows if row["priority"] == "now"}
    claim(now <= {kind.catalog_id for kind in catalog()}, f"every 'now' catalog row has a kind ({sorted(now)})")

    return tuple(results)



def test_every_claim_holds():
    failed = [text for ok, text in claims() if not ok]
    assert not failed, failed


def main() -> int:
    results = claims()
    failed = [text for ok, text in results if not ok]
    for ok, text in results:
        print(("  ok    " if ok else "  FAIL  ") + text)
    print(f"{'PASS' if not failed else 'FAIL'}  test_selfcheck: {len(results) - len(failed)} of {len(results)} claims hold")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
