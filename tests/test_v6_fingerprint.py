"""The v6 fingerprint check can fail, and passes what it should (the checker: tests/_v6_fingerprint.py; why two tiers:
tests/test_equivalence.py).

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_v6_fingerprint.py

- v6's own trajectories pass it (on the tolerance tier, the arrays this machine makes pass it).
- A 1e-6 change to one interior value fails it: a dozen values by hand across the four arrays of four trajectories (the middle
  of the longest stored array among them), and every interior index of every stored array by the size of its weights.
- The arrays numpy makes under AVX2 dispatch (EPYC 7763's arrays, up to 7.1e-12 off v6's) pass it.
- Every v6 digest in tests/test_equivalence.py has a stored fingerprint, and the store holds nothing else.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import types

import numpy as np

import _runner
import _v6_fingerprint as fingerprint
import scenario
import test_equivalence as equivalence
from robot_trajectories import this_week

TESTS = pathlib.Path(__file__).resolve().parent
SRC = TESTS.parent / "src"
PERTURBATION = 1e-6  # WHY 1e-6: a change far below anything a robot could feel, yet ~100x the check's tolerance (see below)


def _week():
    limits, _ = this_week.read_limits(equivalence.EXAMPLE)
    return this_week.week(limits, "can11")


def _with(trajectory, array: str, index: int, delta: float):
    """A copy of ``trajectory``'s four arrays with one value of one array moved by ``delta`` (flat, C-order ``index``)."""
    arrays = {name: np.array(getattr(trajectory, name), dtype=np.float64, order="C") for name in fingerprint.ARRAYS}
    flat = arrays[array].reshape(-1)  # a view: the copy above is contiguous
    assert 0 < index < flat.size - 1, f"{array}[{index}] is not interior (size {flat.size})"
    flat[index] += delta
    return types.SimpleNamespace(**arrays)


def _cases():
    """(label, trajectory, its v6 digest): the scenario's pendulum and five-bar, the week, and the kind with the longest array."""
    return (
        ("pendulum session", scenario.pendulum(), equivalence.V6["pendulum"]),
        ("five-bar session", scenario.fivebar(), equivalence.V6["fivebar"]),
        ("week", _week(), equivalence.V6["week"]),
        ("ur5e/workspace_survey", equivalence._ported("ur5e", "workspace_survey"), equivalence.V6_KINDS["ur5e/workspace_survey"]),
    )


def test_v6s_trajectories_pass():
    failed = {label: problem for label, trajectory, expected in _cases() if (problem := fingerprint.check(trajectory, expected))}
    assert not failed, f"{failed} ({fingerprint.tier()})"


def test_a_1e6_change_to_one_interior_value_fails():
    (_, pendulum, pendulum_digest), (_, fivebar, fivebar_digest), (_, week, week_digest), (_, survey, survey_digest) = _cases()
    longest = np.asarray(survey.positions).size
    perturbations = [
        ("pendulum session", pendulum, pendulum_digest, "t", 1),
        ("pendulum session", pendulum, pendulum_digest, "positions", 13_524),
        ("pendulum session", pendulum, pendulum_digest, "velocities", 27_047),
        ("pendulum session", pendulum, pendulum_digest, "accelerations", 5_003),
        ("five-bar session", fivebar, fivebar_digest, "velocities", 55_475),  # mid-array (sample 27 737, joint 2)
        ("week", week, week_digest, "t", 88_463),
        ("week", week, week_digest, "positions", 1),
        ("week", week, week_digest, "velocities", 123_457),
        ("week", week, week_digest, "accelerations", 176_924),  # the noisiest array under AVX2 (7.1e-12)
        ("ur5e/workspace_survey", survey, survey_digest, "positions", longest // 2),  # the middle of the longest array
        ("ur5e/workspace_survey", survey, survey_digest, "accelerations", 2_000_001),
        ("ur5e/workspace_survey", survey, survey_digest, "t", 437_925),
    ]
    unseen = []
    for label, trajectory, expected, array, index in perturbations:
        for delta in (PERTURBATION, -PERTURBATION):
            if fingerprint.check(_with(trajectory, array, index, delta), expected) is None:
                unseen.append(f"{label} {array}[{index}] {delta:+g}")
    assert not unseen, f"these 1e-6 changes passed the fingerprint check: {unseen}"
    stored_longest = max((int(size) for record in fingerprint.stored().values() for size in record.sizes), default=0)
    assert longest == stored_longest, f"ur5e/workspace_survey's positions ({longest}) are not the longest ({stored_longest})"


def test_every_interior_index_of_every_stored_array_is_seen():
    # A change d to value i moves projection k by w[k, i] * d (exactly, up to one rounding of the product), so it fails the
    # check when max_k |w[k, i]| * d exceeds the tolerance plus the dispatch noise; the noise passes the check (it is below
    # the tolerance), so max_k |w[k, i]| * d > 2 * tolerance is enough. Weights depend only on an array's size: check each.
    sizes = sorted({int(size) for record in fingerprint.stored().values() for size in record.sizes})
    assert sizes, "no stored fingerprints"
    weakest = (np.inf, 0, 0)
    for size in sizes:
        strongest = np.zeros(size)
        for row in fingerprint.weights(size):
            np.maximum(strongest, np.abs(row), out=strongest)
        index = 1 + int(np.argmin(strongest[1:-1]))
        weakest = min(weakest, (float(strongest[index]), size, index))
    weight, size, index = weakest
    assert weight * PERTURBATION > 2 * fingerprint.PROJECTION_ATOL, (
        f"index {index} of a {size}-value array: its largest weight is {weight:.3g}, so a {PERTURBATION:g} change moves a "
        f"projection by only {weight * PERTURBATION:.3g} (tolerance {fingerprint.PROJECTION_ATOL:g})")


CHILD = """
import json, sys
sys.path[:0] = sys.argv[1:3]
import _v6_fingerprint as fingerprint, scenario, test_equivalence as equivalence
from robot_trajectories import this_week
limits, _ = this_week.read_limits(equivalence.EXAMPLE)
cases = {"pendulum session": (scenario.pendulum(), equivalence.V6["pendulum"]),
         "five-bar session": (scenario.fivebar(), equivalence.V6["fivebar"]),
         "UR5e session": (scenario.ur5e(), equivalence.V6["ur5e"]),
         "week": (this_week.week(limits, "can11"), equivalence.V6["week"])}
report = {label: {"bit_for_bit": equivalence.digest(trajectory) == expected, "problem": fingerprint.check(trajectory, expected)}
          for label, (trajectory, expected) in cases.items()}
print(json.dumps({"exact": fingerprint.exact_dispatch(), "tier": fingerprint.tier(), "cases": report}))
"""


def test_avx2_dispatch_arrays_pass():
    if not fingerprint.cpu_features().get("AVX2"):
        _runner.skip("this CPU has no AVX2 (or is not x86): numpy cannot be held to the AVX2 dispatch here")
    environment = {**os.environ, "NPY_ENABLE_CPU_FEATURES": fingerprint.AVX2_FEATURES}
    run = subprocess.run([sys.executable, "-c", CHILD, str(TESTS), str(SRC)], env=environment, capture_output=True, text=True,
                         timeout=900, check=False)
    assert run.returncode == 0, run.stderr[-3000:]
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert not report["exact"], f"NPY_ENABLE_CPU_FEATURES did not take numpy off AVX512_SKX: {report['tier']}"
    failed = {label: case for label, case in report["cases"].items() if case["problem"]}
    assert not failed, f"AVX2-dispatch arrays failed the fingerprint check ({report['tier']}): {failed}"


def test_every_v6_digest_has_a_fingerprint():
    digests, stored = equivalence.v6_digests(), set(fingerprint.stored())
    assert stored == digests, (f"missing {sorted(digests - stored)}; not a v6 digest {sorted(stored - digests)} "
                               "(re-take them: python tests/_v6_fingerprint.py --write)")


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), f"test_v6_fingerprint ({fingerprint.tier()})"))
