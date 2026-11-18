"""This package makes v6's motion, bit for bit: the round's results (its ruler, v6's selfcheck, this week's tapes) hold here.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_equivalence.py

- The vendored kernels are the round's file byte for byte (below the provenance comment).
- The scenario's pendulum session (joined, each segment, its config text), five-bar and UR5e sessions, the week and each of
  its eight tapes equal digests taken from v6 itself: bam_ws docs/trajectory_generator/variants/v6_splice at b5e1f6965, run
  2026-11-18 in the round's venv (Python 3.11.15, numpy 2.3.5). WHY digests: the test then needs nothing from bam_ws.
- ``python -m robot_trajectories.this_week`` writes tapes equal to v6's files array for array and config for config,
  read with ``robot_msgs.load_npz`` (skipped where the round's media is absent; the digests still hold the arrays).
- examples/can11_limits.json is the fixture's PENDULUM_LIMITS, exactly.

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
from robot_trajectories import this_week

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
    "pendulum_config_sha256": "a01b8bf229fb545d38df80a4aa8a31feb76280b3713de08ec74885727f5f1ece",
    "fivebar": "40af2f02f0984d7286b1ac5c7fc6c6356b655672dd2f31e57e16e96e67cc586a",
    "ur5e": "c4cf676051ee5de7f1b2c8c4dc4fd8fe23dcef1216aa52af4992e3a0d3a62394",
    "week": "42b5ed5251db63ac589f6cb9f21feecfc3864995f9a5e008e02fa5ba37cbe7a9",
    "week_config_sha256": "99a216352d630f02b27447074f0efdb5880fe20f4608020203aff94cfe78c5ca",
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
    assert digest(session.trajectory) == V6["pendulum"], "pendulum session"
    assert tuple(digest(segment) for segment in session.segments) == V6["pendulum_segments"], "pendulum segments"
    assert hashlib.sha256(session.to_json().encode()).hexdigest() == V6["pendulum_config_sha256"], "pendulum config text"
    assert digest(scenario.fivebar().trajectory) == V6["fivebar"], "five-bar session"
    assert digest(scenario.ur5e().trajectory) == V6["ur5e"], "UR5e session"


def test_week_is_v6_bit_for_bit():
    limits, _ = this_week.read_limits(EXAMPLE)
    week = this_week.week(limits, "can11")
    assert digest(week.trajectory) == V6["week"], "the week, joined"
    assert hashlib.sha256(week.to_json().encode()).hexdigest() == V6["week_config_sha256"], "the week's config text"
    names = [this_week.tape_name(index, item.record.display_label) for index, item in enumerate(week.steps, start=1)]
    assert names == list(V6["tapes"]), names
    for name, segment in zip(names, week.segments, strict=True):
        assert digest(segment) == V6["tapes"][name], name


def test_this_week_cli_writes_v6s_tapes():
    with tempfile.TemporaryDirectory() as folder:
        assert this_week.main(["--limits", str(EXAMPLE), "--out", folder]) == 0
        out = pathlib.Path(folder)
        for name, expected in V6["tapes"].items():
            assert digest(load_npz(out / f"{name}.trajectory.npz")) == expected, name
        assert digest(load_npz(out / "week.trajectory.npz")) == V6["week"], "week.trajectory.npz"
        if not V6_TAPES.is_dir():
            _runner.skip(f"v6's tape files are not here ({V6_TAPES}); the digests above held")
        for name in (*V6["tapes"], "week"):
            mine, theirs = load_npz(out / f"{name}.trajectory.npz"), load_npz(V6_TAPES / f"{name}.trajectory.npz")
            for array in ARRAYS:
                assert np.array_equal(getattr(mine, array), getattr(theirs, array)), (name, array)
            assert mine.joint_names == theirs.joint_names, name
            assert (out / f"{name}.config.json").read_text() == (V6_TAPES / f"{name}.config.json").read_text(), f"{name}.config.json"


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_equivalence (v6, bit for bit)"))
