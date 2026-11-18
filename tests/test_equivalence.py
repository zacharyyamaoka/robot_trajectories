"""The signal library makes v6's motion, bit for bit: the round's results (its ruler, v6's claims, this week's tapes) hold here.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_equivalence.py

- The vendored kernels are the round's file byte for byte (below the provenance comment).
- The scenario's pendulum session (joined and each segment), five-bar and UR5e sessions, the week and each of its eight
  tapes equal digests taken from v6 itself: bam_ws docs/trajectory_generator/variants/v6_splice at b5e1f6965, run 2026-11-18
  in the round's venv (Python 3.11.15, numpy 2.3.5). WHY digests: the test then needs nothing from bam_ws.
- Every ported kind (defaults, one joint where it takes one) and every higher-order kind (v6's plans), on all three robots,
  equals v6's record or plan: 63 digests taken from robot_trajectories 0.0.1 (main 138fc43) on 2026-11-18, before the rewrite.
- ``python -m robot_trajectories.this_week`` writes tapes equal to v6's files array for array (skipped where the round's
  media is absent; the digests still hold the arrays).
- examples/can11_limits.json is the fixture's PENDULUM_LIMITS, exactly.

Not held any more, deliberately (0.1.0, PEP 0002): v6's config TEXT (the pendulum's and the week's ``config.json`` digests and
the tapes' ``.config.json`` files). The configuration is now a recipe of calls (``.recipe.json``), a new format; what it makes
is held above, sample for sample, and every recipe here is held to rebuild its trajectory exactly.

A deliberate change to the motion changes these digests: re-derive them and say why in the commit message.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import pathlib
import tempfile

import numpy as np
from robot_msgs import load_npz

import _runner
import scenario
from fixtures import robots
from robot_trajectories import chain, kinds, labels, rebuild, recipe, speed_ladder, split, this_week

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "can11_limits.json"
KERNELS = ROOT / "src" / "robot_trajectories" / "kernels.py"
V6_TAPES = pathlib.Path(os.environ.get("V6_TAPES", "/home/bam/bam_ws/reports/media/trajectory-generator-2026-11-18/v6/tapes"))

# sha256 of bam_ws docs/trajectory_generator/shared/kernels.py at b5e1f6965
KERNELS_SHA256 = "9485aa020f3ec0242045ae385a90b02421b332b6d92e474c76df30b685b96f11"

# sha256 over t, positions, velocities, accelerations (float64, in that order), from v6 at b5e1f6965
V6 = {
    "pendulum": "ed8f523b3479e4ceea5b1992fe86523b57be6a51dbd816f411057d32a2cc29d4",
    "pendulum_segments": (
        "cc8b9e2ea402398fbb5046266c86ee1afa7422b4921b3aeea212e9d9b68167cf",
        "cab5a50a5218327c453110262dfd1306cbbc7c4e1d545c6f3b4e410f0ff84153",
        "05f0b74c89eb5bdffe1a46e08414437e82a3f61f72772326c49d95b81fe77e8b",
        "f178ce2de3e3b3d7b17be9154b1ea3624a1120d55643fead254cf940cf20dc00",
        "8ce71482d8803e782127312d82d94b492c293b2f9a11cadf8b790fd12a5c732e",
        "e16ebe86d07f85cca6f9c8f667265f644dd8127161f3d198c6c20a9d692f3353",
        "e110f650098ce6bfa3e4885985e71a28bf70d63261566e5420de9a9472c1d584",
    ),
    "fivebar": "40af2f02f0984d7286b1ac5c7fc6c6356b655672dd2f31e57e16e96e67cc586a",
    "ur5e": "c4cf676051ee5de7f1b2c8c4dc4fd8fe23dcef1216aa52af4992e3a0d3a62394",
    "week": "42b5ed5251db63ac589f6cb9f21feecfc3864995f9a5e008e02fa5ba37cbe7a9",
    "tapes": {
        "01-settle": "cab5a50a5218327c453110262dfd1306cbbc7c4e1d545c6f3b4e410f0ff84153",
        "02-repeat-floor": "8abc53baee624cd3f475aae7312d30891abe3b7b8e291ac3e8de018c222990c1",
        "03-linear-chirp": "3ce8cb1cadcf577fe7d9464be6670c12c9e1d2ace85bc15574a75bc8376e39ba",
        "04-velocity-plateaus": "115cc84ff6b460728c907e7f799d59f6169c8b087dadfade068dd9f670633226",
        "05-step-train": "22850b11ee582296df21d951f0630396b8323d0d532c4f4d2b4e74430a8fcc74",
        "06-speed-ladder": "cd22bed85884188b694b000dd696f039cffa04d4311c7282e1913bb49ea095e7",
        "07-deployment-classes": "363ccb801b22bfc555dd61e786caaf357eead4fb12205d3ee65c22f2192ba894",
        "08-deployment-classes-held-out": "1532b0c628151bee098ec0eae6c73855327f91b2d14810eb8a0033e87d67c4c5",
    },
}
ARRAYS = ("t", "positions", "velocities", "accelerations")


def digest(trajectory) -> str:
    hasher = hashlib.sha256()
    for name in ARRAYS:
        hasher.update(np.ascontiguousarray(getattr(trajectory, name), dtype=np.float64).tobytes())
    return hasher.hexdigest()


def test_vendored_kernels_are_the_rounds_byte_for_byte():
    lines = KERNELS.read_text(encoding="utf-8").splitlines(keepends=True)
    body = "".join(itertools.dropwhile(lambda line: line.startswith("#"), lines))
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == KERNELS_SHA256, "kernels.py differs from the round's file"


def test_example_limits_are_the_fixtures():
    limits, _ = this_week.read_limits(EXAMPLE)
    fixture = robots.PENDULUM_LIMITS
    assert limits.joint_names == fixture.joint_names and limits.position_kind == fixture.position_kind
    for bound in ("min_position", "max_position", "max_velocity", "max_acceleration", "max_jerk", "max_effort"):
        mine, theirs = getattr(limits, bound), getattr(fixture, bound)
        assert (mine is None and theirs is None) or np.array_equal(mine, theirs), bound


def test_scenario_is_v6_bit_for_bit():
    session = scenario.pendulum()
    assert digest(session) == V6["pendulum"], "pendulum session"
    assert tuple(digest(segment) for segment in split(session)) == V6["pendulum_segments"], "pendulum segments"
    assert digest(rebuild(recipe(session))) == V6["pendulum"], "the pendulum session rebuilt from its recipe"
    assert digest(scenario.fivebar()) == V6["fivebar"], "five-bar session"
    assert digest(scenario.ur5e()) == V6["ur5e"], "UR5e session"


def test_week_is_v6_bit_for_bit():
    limits, _ = this_week.read_limits(EXAMPLE)
    week = this_week.week(limits, "can11")
    assert digest(week) == V6["week"], "the week, joined"
    names = [this_week.tape_name(index, label) for index, label in enumerate(labels(week), start=1)]
    assert names == list(V6["tapes"]), names
    for name, segment in zip(names, split(week), strict=True):
        assert digest(segment) == V6["tapes"][name], name
        assert digest(rebuild(segment)) == V6["tapes"][name], f"{name} rebuilt from its own recipe"
    assert digest(chain(*split(week))) == V6["week"], "chain(*split(week)) is the week"


# v6 at robot_trajectories 0.0.1 (138fc43): every record class with its defaults (one joint where it takes one; the floor tape
# for Repeat; the five-bar ladder without its 5° margins) and every plan, joined, on the three robots of the scenario.
V6_KINDS = {
    "pendulum/chirp": "05f0b74c89eb5bdffe1a46e08414437e82a3f61f72772326c49d95b81fe77e8b",
    "pendulum/deployment_motion_classes": "363ccb801b22bfc555dd61e786caaf357eead4fb12205d3ee65c22f2192ba894",
    "pendulum/hold": "f178ce2de3e3b3d7b17be9154b1ea3624a1120d55643fead254cf940cf20dc00",
    "pendulum/home": "832e3d6c491beca727b84ba250ebf8fe58f624d0d2a0471d9527c944f043c446",
    "pendulum/linear_chirp": "3ce8cb1cadcf577fe7d9464be6670c12c9e1d2ace85bc15574a75bc8376e39ba",
    "pendulum/move": "d3b16d0242415f03c677fd30b51c9170667ebb3315f3408389c54e6b78465c43",
    "pendulum/multisine": "2104808a59bf4bc18d89f4371b5f6ce501eac72b97a804e02d5ec5fdf16ac498",
    "pendulum/random_waypoints": "79da1efe2a1dcaa0c781d27cab656eaafdf2edb35c49c603b9281946cd1fd558",
    "pendulum/repeat_floor_tape": "8abc53baee624cd3f475aae7312d30891abe3b7b8e291ac3e8de018c222990c1",
    "pendulum/sine_dwell": "96d10d2ebae7c3a09938061cc30a8759a339630b518d53c71fa63f4e09fc6a3b",
    "pendulum/speed_ladder": "cd22bed85884188b694b000dd696f039cffa04d4311c7282e1913bb49ea095e7",
    "pendulum/step_train": "22850b11ee582296df21d951f0630396b8323d0d532c4f4d2b4e74430a8fcc74",
    "pendulum/velocity_plateaus": "115cc84ff6b460728c907e7f799d59f6169c8b087dadfade068dd9f670633226",
    "pendulum/velocity_sweep": "8ce71482d8803e782127312d82d94b492c293b2f9a11cadf8b790fd12a5c732e",
    "pendulum/workspace_sweep": "cce44a6822dcb17cdb521bd3dcd1909f8e7b7a7b849766d66d7348f13f358f8b",
    "fivebar/chirp": "d94df85f5a9bdba068cfd75f27d9def7e5ac9cf9e81694de382279edf3391f82",
    "fivebar/deployment_motion_classes": "b847cd6393a9d2a781a64603056c5ec679c50deb540e14d42d13b3afbc86924a",
    "fivebar/hold": "3449ff884b36ec4e9ae988ddaf3c85e709e0075ae573fa08c27a4b85d99f4489",
    "fivebar/home": "de816be9f0205783d32e4693749cca58552ad2e5e51faf0a3319dc754734e09a",
    "fivebar/linear_chirp": "caf95bdd765ed9ca31c961314de1239f5bb9b38254b257f77dc8f61fa6fe7160",
    "fivebar/move": "48b91b84105faa6521b15c1592f0d1781398d476f0fc554c08ce8ce67463e9d9",
    "fivebar/multisine": "efbea6029b35f843fb918b56e5667026d9f2157582bd37331253e9928a16c100",
    "fivebar/random_waypoints": "fa927ee1fd4697decc4d8918fb33017b4d5b0d16b1b29ca55f6935616a6fdb36",
    "fivebar/repeat_floor_tape": "9fe968b2419becbf7e5d6096a137fe5dfb973c7a30979e6b5eb136f310448d8d",
    "fivebar/sine_dwell": "2c3fb40ae616eb2ce1d7a537e9f6cb3ebcdf718fc4b5007a1b9b670d2136b513",
    "fivebar/speed_ladder": "3888b184463db23f306f21f0a5536fbb75a15e6415acbfb62edeb7ec763fba9f",
    "fivebar/step_train": "91c8897eb66018ac8142cefd38558583e553e2c2c026f8b1e20f1c2f9ca75fe7",
    "fivebar/velocity_plateaus": "0e2ed11a5a9c00a6cefa9d5454d81c47ea2d0c40a37dd4b585b6358c467dc7a2",
    "fivebar/velocity_sweep": "8e8fdaf82f2b414fd439ad2079e44c6f25836767fa241156fbb130e5a64d69d6",
    "fivebar/workspace_sweep": "1d204c0376e131a7d459dae1727790391424a1edc3dd02479d52c4a810a27ffc",
    "ur5e/chirp": "31e21bb71b5ac150ac07316988076b5e2417b69e1d6ad8b24cb13b1761b1226d",
    "ur5e/deployment_motion_classes": "f9ae6b1af30ec143ef584564ac3b6479c6463e29a506e86b3a0c382d997179fc",
    "ur5e/hold": "0419021d62e2168663ca3b4e46b3da853a0c05b5c6d80102f261c6bee2dfd17b",
    "ur5e/home": "e66fb8cb5dca78058207f07f7ca2c3835d9324fb35a995f4478a7d45a2915e50",
    "ur5e/linear_chirp": "49d8c642d1d9fffbf85551996db589bb6946e4fa60ea744ea40e7342c64ef659",
    "ur5e/move": "36d0af8ecda1a87e1f4ab4097227b2375ab8ab1d3733e817c80b030e53e530b5",
    "ur5e/multisine": "613f008cff9a3bd203a4ae62ad8a2dfcfac30663d382d4853fa9b17c1ad64209",
    "ur5e/random_waypoints": "eab7eed9490010a36aa45698e1547356ab771ba61f7dad215a23afc178050679",
    "ur5e/repeat_floor_tape": "7834b72366b592b7acc99cd7a640af2a518757c865e1ba59a84bdf395044acd2",
    "ur5e/sine_dwell": "9b3279244dd0115ad2ac55381728e64a7daff0232c46b299b958af7a132a4b6d",
    "ur5e/speed_ladder": "8e01f4459898796bee5854a9d64251e94a71c6fa88e15bf53ab41e8f9559db50",
    "ur5e/step_train": "9043ab0f70058d27cb9bb6f7ed18a2d238da3788a410aaeb769bf82a22eab774",
    "ur5e/velocity_plateaus": "d174b26a5034ff7a3b29071521b40a21a3c4c10ad082a7cd5e30a75f4d8e6dcd",
    "ur5e/velocity_sweep": "b71ebd89b4e1cfc35a74d11ccbde0652038f988b566439a40832ee5d0ba4f374",
    "ur5e/workspace_sweep": "32fd292450023e2443cc812d86da0fa9d27e87c2b825774cbad38b3a8d597c84",
    "pendulum/pid_tuning": "e35ecd2aadd9c9df239e6621e92a04883247dfb2f5986f585283b01837d3bcfa",
    "pendulum/friction_id": "8ce71482d8803e782127312d82d94b492c293b2f9a11cadf8b790fd12a5c732e",
    "pendulum/tracking_check": "556514f4054e44d00b0ba9e30f3d5e6a2b63f62aa483f72aed07d48097741d13",
    "pendulum/workspace_survey": "af005b662d8fc10fd53bcc1ab7691128bca4f1db482632cb55963e0f95ce3460",
    "pendulum/dynamics_excitation": "2104808a59bf4bc18d89f4371b5f6ce501eac72b97a804e02d5ec5fdf16ac498",
    "pendulum/sim_to_real_tracking": "42b5ed5251db63ac589f6cb9f21feecfc3864995f9a5e008e02fa5ba37cbe7a9",
    "fivebar/pid_tuning": "7b474d393cd2e30ecc8b64b7b4f36af90b29c9853f092679b65fe35367898bd4",
    "fivebar/friction_id": "8e8fdaf82f2b414fd439ad2079e44c6f25836767fa241156fbb130e5a64d69d6",
    "fivebar/tracking_check": "18f475d98f7c51d781fb3a437d6d9076984f2013dd37e13ff75ba559642acff6",
    "fivebar/workspace_survey": "9fcefae291b9a3db440abc4618d4d70bd89522ef2c563a5344dccea086569fc5",
    "fivebar/dynamics_excitation": "efbea6029b35f843fb918b56e5667026d9f2157582bd37331253e9928a16c100",
    "fivebar/sim_to_real_tracking": "b806714335f175ceaf6a7d2bb1a4d5d5302c5a59f3c05481d663d193d1d95a66",
    "ur5e/pid_tuning": "91506cdb4a037a04d350b4151baaeb9afab16bcd2c06801e13dab1fea1ac3a58",
    "ur5e/friction_id": "b71ebd89b4e1cfc35a74d11ccbde0652038f988b566439a40832ee5d0ba4f374",
    "ur5e/tracking_check": "d47fcc8f18ebd9a2fa6e032435d9add6fc45d452de8ec8d8a7392499e9c3e607",
    "ur5e/workspace_survey": "5a412c3e1fae9199ad4aee35222d439f10638876ece57c45536c8804481ac6ac",
    "ur5e/dynamics_excitation": "613f008cff9a3bd203a4ae62ad8a2dfcfac30663d382d4853fa9b17c1ad64209",
    "ur5e/sim_to_real_tracking": "e172ed5f78c5053d1a9c3705cce42192fd7245e15073b212e8068a9c857038a6",
}
ROBOTS = {"pendulum": scenario.pendulum_space, "fivebar": scenario.fivebar_space, "ur5e": scenario.ur5e_space}
JOINT_OF = {"pendulum": "can11", "fivebar": "j2", "ur5e": "elbow_joint"}


def _ported(robot: str, name: str):
    space, joint = ROBOTS[robot](), JOINT_OF[robot]
    if name == "speed_ladder" and robot == "fivebar":
        return speed_ladder(space, joint, margin_rad=0.0, margin_fraction=0.05)  # its hole-free box is 6.3° wide: no 5° margins
    takes_joint = "joint" in kinds()[name].parameters
    return kinds()[name].function(space, joint) if takes_joint else kinds()[name].function(space)


def test_every_ported_kind_and_plan_is_v6_on_every_robot():
    different = [key for key, expected in V6_KINDS.items() if digest(_ported(*key.split("/"))) != expected]
    assert not different, different


def test_this_week_cli_writes_v6s_tapes():
    with tempfile.TemporaryDirectory() as folder:
        assert this_week.main(["--limits", str(EXAMPLE), "--out", folder]) == 0
        out = pathlib.Path(folder)
        for name, expected in V6["tapes"].items():
            assert digest(load_npz(out / f"{name}.trajectory.npz")) == expected, name
        assert digest(load_npz(out / "week.trajectory.npz")) == V6["week"], "week.trajectory.npz"
        for name in V6["tapes"]:
            assert digest(rebuild((out / f"{name}.recipe.json").read_text())) == V6["tapes"][name], f"{name}.recipe.json rebuilds it"
        if not V6_TAPES.is_dir():
            _runner.skip(f"v6's tape files are not here ({V6_TAPES}); the digests above held")
        for name in (*V6["tapes"], "week"):
            mine, theirs = load_npz(out / f"{name}.trajectory.npz"), load_npz(V6_TAPES / f"{name}.trajectory.npz")
            for array in ARRAYS:
                assert np.array_equal(getattr(mine, array), getattr(theirs, array)), (name, array)
            assert mine.joint_names == theirs.joint_names, name


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_equivalence (v6, bit for bit)"))
