"""v6's own claims, carried into the signal library (beyond the round's ruler), plus the claims the new contract makes.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_selfcheck.py

Ported from robot_trajectories 0.0.1's selfcheck (itself bam_ws docs/trajectory_generator/variants/v6_splice/selfcheck.py at
b5e1f6965): the same subjects, in the new API (records and plans became functions; the workspace became the JointSpace).
As a script it prints one line per claim.
"""

from __future__ import annotations

import functools
import importlib
import random

import numpy as np

import scenario
from fixtures import reference, robots
from robot_trajectories import (JointSpace, MissingRung, WaypointTrajectoryGenerator, bands, catalog, kinds, labels, move, notes,
                                pid_tuning, rebuild, recipe, repeat, segment_info, speed_ladder, split, trajectory_catalog)

CARTESIAN = {"cartesian_path", "cartesian_waypoints", "pick_and_place"}


@functools.lru_cache(maxsize=1)
def claims() -> tuple[tuple[bool, str], ...]:
    """Every claim, computed once: (holds, text)."""
    results: list[tuple[bool, str]] = []

    def claim(ok: bool, text: str) -> None:
        results.append((bool(ok), text))

    def inside(trajectory, limits, info=None) -> str:
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
        # WHY per end (2026-11-18, lanes A-F): a kind may declare a moving start/end (a synchronised hand-over) or an
        # event-ended finish (ends="anywhere"); only the ends it declares at rest are held to rest here.
        rest_ends = [index for index, declared in ((0, getattr(info, "starts", "at_rest")), (-1, getattr(info, "ends", "at_rest")))
                     if declared == "at_rest"]
        if rest_ends and np.max(np.abs(qd[rest_ends])) > 1e-6:
            problems.append("not at rest at the ends")
        return ", ".join(problems)

    # 1. the ruler's motion is the reference's, sample for sample ----------------------------------------------------------------
    ours = scenario.pendulum()
    theirs, _, _ = reference.pendulum_session(reference.PENDULUM_CONFIG)
    claim(np.array_equal(ours.positions, theirs.positions) and np.array_equal(ours.t, theirs.t), "pendulum session = the bare-kernel reference, every sample")
    claim(np.array_equal(scenario.ur5e().positions, reference.ur5e_excitation().positions), "UR5e session = the reference, every sample")

    # 2. every kind, defaults, on every robot: inside the limits, at rest at both ends, five-bar reachable, rebuilt exactly -------
    spaces = {"pendulum": scenario.pendulum_space(), "five-bar": scenario.fivebar_space(), "UR5e": scenario.ur5e_space()}
    joint_of = {"pendulum": "can11", "five-bar": "j2", "UR5e": "elbow_joint"}
    for robot, space in spaces.items():
        for name, info in kinds().items():
            # WHY: test files register deliberately broken kinds (tests/test_contract.py's contract_too_fast);
            # in one pytest process they share the registry, so check shipped and plugin kinds only.
            if info.module.split(".")[0] in ("tests", "conftest") or info.module.split(".")[-1].startswith("test_"):
                continue
            joint = joint_of[robot]
            if name in CARTESIAN:
                try:
                    info.function(space, [space.home], ) if name != "pick_and_place" else info.function(space, (0, 0, 0), (0, 0, 0))
                    claim(False, f"{robot}: {name} is refused without kinematics")
                except MissingRung as error:
                    claim("with_kinematics" in str(error), f"{robot}: {name} is refused without kinematics, naming with_kinematics")
                continue
            arguments, options = [], {}
            if "joint" in info.parameters:
                arguments = [joint]
            if name == "joint_waypoints":
                half = 0.3 * (space.room(space.home).high - space.home)
                arguments = [[space.home + half, space.home - half]]
            if name == "speed_ladder" and robot == "five-bar":
                options = {"margin_rad": 0.0, "margin_fraction": 0.05}  # its hole-free box is 6.3° wide: no 5° margins
            try:
                trajectory = info.function(space, *arguments, **options)
            except MissingRung as error:
                # WHY (2026-11-18): a kind that needs a rung the plain robot lacks (kinematics, dynamics, torque limits, a
                # named home) must refuse and name the call that adds it; its motion is tested in its lane's own file.
                claim(any(word in str(error) for word in ("with_", "Add it", "pass ", "Pass ")), f"{robot}: {name} is refused for a missing input, naming the fix")
                continue
            except ValueError as error:
                claim(False, f"{robot}: {name} generates ({error})")
                continue
            problem = inside(trajectory, space.limits, info)
            if robot == "five-bar" and not problem:
                off = [q for q in trajectory.positions[::5] if (p := robots.fivebar_passive(*q)) is None or p[2] < 0.2]
                problem = f"leaves the reachable set at {len(off)} samples" if off else ""
            claim(not problem, f"{robot}: {name} inside every limit, at rest at both ends{' (' + problem + ')' if problem else ''}")
            again = rebuild(trajectory, space)
            claim(np.array_equal(again.positions, trajectory.positions), f"{robot}: {name} rebuilds from its recipe, sample for sample")

    # 3. the speed ladder is one path under several waypoint trajectory generators; the acceleration clamp is reported -------------------
    ladder_module = importlib.import_module("robot_trajectories.kinds.speed_ladder")
    claim([law.peak_speed_fraction for law in ladder_module.waypoint_trajectory_generators()] == [0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
          "speed_ladder: one WaypointTrajectoryGenerator per rung")
    pendulum = spaces["pendulum"]
    ladder = speed_ladder(pendulum, "can11")
    legs = [band for band in bands(ladder) if band.label.endswith("A to B")]
    claim(len(legs) == 6 * 7, f"ladder: 6 rungs x (5 repeats + 2 held out) A-to-B legs ({len(legs)})")
    spans = {round(band.end_s - band.start_s, 6) for band in legs if "repeat" in band.label and band.label.startswith("speed ladder: 20 %")}
    claim(len(spans) == 1, f"ladder: the 20 % rung's 5 repeats are the same input (leg durations {sorted(spans)})")
    held = [band.label for band in legs if "held out" in band.label]
    claim(held[0].endswith("(seed 0): A to B") and "seed 51" in held[-1], f"ladder: held-out seeds are round x 1000 + rung x 10 + k ({held[0]!r} ... {held[-1]!r})")
    claim(any("acceleration-bound" in note for note in notes(ladder)), "ladder: rungs above what the acceleration allows are noted")
    timing = WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=1.0).leg_timing(np.array([-0.24]), np.array([0.24]),
                                                                                   pendulum.limits)
    claim(timing.bound == "acceleration" and timing.peak_speed_fraction < 0.8,
          f"WaypointTrajectoryGenerator clamps to the acceleration limit ({timing})")
    try:
        WaypointTrajectoryGenerator("ruckig", peak_speed_fraction=0.5)
        claim(False, "a waypoint trajectory generator with no backend installed is refused")
    except ValueError as error:
        claim("robot_trajectories.waypoint_trajectory_generators" in str(error) and "lane E" in str(error),
              "a waypoint trajectory generator with no backend is refused, naming the entry-point group to install one under")
    trapezoid = move(pendulum, "can11", waypoint_generator=WaypointTrajectoryGenerator.trapezoid(peak_speed_fraction=0.8))
    claim(not inside(trapezoid, pendulum.limits),
          "a trapezoid waypoint trajectory generator runs inside every limit (built in since 0.1.0)")

    # 4. the constant-speed chirp: bigger at low frequency than the shared one -------------------------------------------------------
    week_chirp = kinds()["linear_chirp"].function(pendulum, "can11", f_start_hz=0.2, f_end_hz=5.0, hold_after_s=0.0)
    shared = split(ours)[2]
    claim(np.ptp(week_chirp.positions) > 20 * np.ptp(shared.positions), f"constant-speed chirp swings {np.ptp(week_chirp.positions) / np.ptp(shared.positions):.0f}x the shared chirp")
    claim(not inside(week_chirp, pendulum.limits), "constant-speed chirp inside every limit")

    # 5. repeat: every play is the same input, from the same place ------------------------------------------------------------------
    tape = move(pendulum, "can11", targets_fraction=(0.6,), back_to_start=False)  # ends away from its start
    played = repeat(tape, 3)
    backs = [band for band in bands(played) if band.label.endswith("back to start")]
    claim(len(backs) == 2, f"repeat moves back to the start between plays when the tape ends elsewhere ({len(backs)} moves)")
    claim(len(segment_info(played)) == 1 and np.array_equal(rebuild(played).positions, played.positions), "repeat is one segment, and rebuilds from its recipe")

    # 6. deployment classes: PID tuna's draws; the sealed seeds -----------------------------------------------------------------------
    classes = importlib.import_module("robot_trajectories.kinds.deployment_motion_classes")
    claim(classes.HELD_OUT_SEEDS == (3, 4, 5) and classes.TRAIN_SEEDS == (0, 1, 2), "deployment classes: train seeds 0-2, held out 3-5")
    drawn = classes.legs("pickplace", random.Random(0))
    claim(len(drawn) == 5 and all(0.4 <= abs(target) <= 0.95 for target, _, _ in drawn[:4]), "deployment classes: pickplace is 4 targets at 40-95 % of the room + home")

    # 7. higher-order kinds: one call in the recipe, each segment its own call, the last one waits, any robot -----------------------
    tuning = pid_tuning(pendulum, "can11")
    import json
    document = json.loads(recipe(tuning))
    claim([call["kind"] for call in document["calls"]] == ["pid_tuning"], "a higher-order kind's recipe is its own one call")
    claim([entry["call"]["kind"] for entry in segment_info(tuning)] == ["step_train", "hold", "chirp", "hold"], "its segments keep the calls that make each alone")
    claim([entry.get("wait_after", False) for entry in segment_info(tuning)] == [False, False, False, True], "its last segment says wait")
    claim(all(np.array_equal(rebuild(piece).positions, piece.positions) for piece in split(tuning)), "every split piece rebuilds alone from its own call")
    five = pid_tuning(spaces["five-bar"], "j2")
    claim(len(segment_info(five)) == 4, "the same higher-order kind runs on another robot (pid_tuning on the five-bar)")
    claim(labels(scenario.pendulum()) == ["step train", "hold", "chirp", "hold", "velocity sweep", "random waypoints", "home"], "chained higher-order kinds flatten into their segments")

    # 8. the ladder of inputs: refusals name the fix; slots unlock catalog rows; a different space is refused ------------------------
    try:
        JointSpace.from_description(robots.UR5E_DESCRIPTION)
        claim(False, "a description without acceleration is refused")
    except MissingRung as error:
        claim("max_acceleration=" in str(error), "a description without acceleration is refused, naming max_acceleration=")
    arm = scenario.ur5e_space()
    unlocked = [sum(entry.status.state != "locked" for entry in catalog(space=space)) for space in
                (arm, arm.with_kinematics(robots.UR5eKinematics()), arm.with_kinematics(robots.UR5eKinematics()).with_dynamics(robots.PendulumDynamics()))]
    claim(unlocked[0] < unlocked[1] < unlocked[2], f"filling the slots unlocks catalog entries: {unlocked[0]} -> {unlocked[1]} (+ kinematics) -> {unlocked[2]} (+ dynamics) of {len(catalog())}")
    with_kinematics = arm.with_kinematics(robots.UR5eKinematics())
    claim(catalog(space=with_kinematics)["workspace_sweep"].status.state == "fallback", "kinematics alone builds no reachable set: workspace_sweep still falls back")
    claim(catalog(space=pendulum)["hold"].status.state == "available", "hold is available from bare joint limits")
    try:
        rebuild(recipe(ours), JointSpace.from_limits(robots.PENDULUM_LIMITS, max_acceleration=5.0))
        claim(False, "a recipe is refused on other limits")
    except ValueError as error:
        claim("another space" in str(error), "a recipe is refused on other limits (the space's fingerprint)")
    claim(np.array_equal(rebuild(ours).positions, ours.positions), "rebuild(logged trajectory) remakes it (its header carries the recipe)")

    # 9. adding a kind is one file: found, tagged, mapped, with no other edit ------------------------------------------------------------
    rows = trajectory_catalog.rows()
    claim("sine_dwell" in catalog(use_case="bandwidth").implemented.names and kinds()["sine_dwell"].catalog_id == "faded_sine",
          "sine_dwell (one file) is registered, tagged bandwidth, mapped to catalog row faded_sine")
    claimed = [info.catalog_id for info in kinds().values() if info.catalog_id]
    claim(len(claimed) == len(set(claimed)), "no catalog row is claimed by two kinds")
    claim(set(claimed) <= {row["id"] for row in rows}, "every claimed catalog row exists in catalog.json")
    now = {row["id"] for row in rows if row["priority"] == "now"}
    claim(now <= set(claimed), f"every 'now' catalog row has a kind ({sorted(now)})")
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
