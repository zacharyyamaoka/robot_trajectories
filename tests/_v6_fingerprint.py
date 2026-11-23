"""v6's motion, fingerprinted: a check against v6 that holds on every CPU, beside tests/test_equivalence.py's sha256 digests,
which hold only where numpy dispatches the way v6's machine did.

    PYTHONPATH=src /home/bam/robot_trajectories/.venv/bin/python tests/_v6_fingerprint.py --check   # vs the store, worst noise
    PYTHONPATH=src /home/bam/robot_trajectories/.venv/bin/python tests/_v6_fingerprint.py --write   # re-take the store

A fingerprint, per array (t, positions, velocities, accelerations, each flattened in C order): its length and size (held
exactly), its first and last value, and its dot product with each of 16 rows of fixed-seed Gaussian weights (``weights``),
which are regenerated from the seed at check time and never stored. ``--write`` takes the fingerprints from v6's arrays
bit for bit only: it refuses unless numpy dispatches AVX512_SKX and every v6 digest in tests/test_equivalence.py matches.

WHY two tiers (diagnosis 2026-11-23, branch claude/diag-equivalence, diag/dump_equivalence.py): on GitHub's runners the
digests held on 6 of 6 AVX-512 machines (EPYC 9V74 and 9V45, Xeon 8573C and 8370C; Python 3.11-3.13; numpy 2.4.6 and 2.5.5)
and failed on 3 of 3 EPYC 7763 (Zen 3, no AVX-512), always on the same 29,520 of 1.74 M values: numpy picks its SIMD kernels
for sin/cos/exp at run time, and the AVX2 kernels round differently from the AVX512_SKX ones in the last bits. Worst
difference 7.1e-12 (the week's accelerations); positions <= 7.8e-16, velocities <= 1.4e-14. This machine (Ryzen 9 7900X,
AVX-512) makes the EPYC 7763's arrays bit for bit under NPY_ENABLE_CPU_FEATURES=AVX2_FEATURES below.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import pathlib
import platform
from collections.abc import Iterator
from functools import cache
from typing import NamedTuple

import numpy as np

ARRAYS = ("t", "positions", "velocities", "accelerations")
STORE = pathlib.Path(__file__).with_name("v6_fingerprints.npz")
# numpy's x86-64 features up to AVX2, none of AVX-512: what an EPYC 7763 dispatches. CI's second run of test_equivalence.py
# sets the same string (.github/workflows/test.yml), so the tolerance tier runs on every push whatever the runner's CPU.
AVX2_FEATURES = "SSE SSE2 SSE3 SSSE3 SSE41 POPCNT SSE42 AVX F16C FMA3 AVX2"

SEED = 20261118  # the round's date; any fixed seed would do, but changing it invalidates the store
PROJECTIONS = 16

# WHY these tolerances (measured 2026-11-23 on this machine: v6's arrays vs its AVX2-dispatch arrays, which are the EPYC
# 7763 runners' arrays bit for bit; `--check` under NPY_ENABLE_CPU_FEATURES=AVX2_FEATURES walks all 92 trajectories the tests
# hold to v6, 68 of which are then not bit for bit):
# - First and last value: |mine - v6| <= 1e-8 + 1e-9 * |v6|. The worst value noise measured (the 1.74 M values of the three
#   sessions and the week) is 7.1e-12, >= 1400x margin; at the endpoints of all 92 it is 5.1e-16.
# - Each projection: |mine - v6| <= 1e-8, absolute, with no relative term: a projection is about as large as its array's
#   norm (up to 9e4, the week's t), so any relative term >= 1e-11 would hide a 1e-6 change. Dispatch noise moves a
#   projection by sum_i w_i * dx_i ~ N(0, sum_i dx_i^2); sqrt(sum dx^2) is 6.1e-11 for the noisiest array measured (the
#   week's accelerations), and the worst projection moved by 1.4e-10 over all 92 (fivebar/sim_to_real_tracking): 73x
#   margin. The CI runners' own EPYC 7763 arrays move it by 1.1e-10.
#   The other side: a change d to one value i moves projection k by w[k, i] * d, and over every interior index of every
#   stored array (88 sizes, up to 2.6 M values) the largest of the 16 weights is >= 0.41, so a 1e-6 change moves some
#   projection by >= 4.1e-7, 41x the tolerance. The tolerance sits near the geometric middle of the noise (1.4e-10) and
#   the smallest signal (4.1e-7); tests/test_v6_fingerprint.py asserts both sides.
# - Projections are summed exactly (math.fsum of the products; an IEEE product is correctly rounded on every SIMD path), so
#   no BLAS or SIMD summation order adds noise of its own: what moves a projection is what moved the values.
VALUE_ATOL, VALUE_RTOL = 1e-8, 1e-9
PROJECTION_ATOL = 1e-8


class Fingerprint(NamedTuple):
    lengths: tuple[int, ...]  # len(array) per array, in ARRAYS order
    sizes: tuple[int, ...]  # array.size per array
    values: np.ndarray  # (len(ARRAYS), 2 + PROJECTIONS): first value, last value, the projections


def cpu_features() -> dict[str, bool]:
    """numpy's run-time CPU features: what its SIMD dispatch may use on this machine, after NPY_ENABLE_CPU_FEATURES."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features
    except ImportError:  # numpy < 2
        from numpy.core._multiarray_umath import __cpu_features__ as features
    return dict(features)


def exact_dispatch() -> bool:
    """True where numpy dispatches its AVX512_SKX kernels, as v6's machine did: there v6's digests hold bit for bit.

    WHY (diagnosis 2026-11-23, see the module docstring): 6 of 6 AVX-512 GitHub runners matched v6 bit for bit; 3 of 3
    EPYC 7763 (AVX2 only) differed in the same 29,520 of 1.74 M values, by up to 7.1e-12 (positions <= 7.8e-16), and this
    AVX-512 machine reproduces those arrays exactly once NPY_ENABLE_CPU_FEATURES turns AVX512_SKX off.
    """
    return bool(cpu_features().get("AVX512_SKX"))


def tier() -> str:
    """Which tier runs here, in words, for failure messages."""
    features = cpu_features()
    simd = "AVX512_SKX" if features.get("AVX512_SKX") else "AVX2" if features.get("AVX2") else "no AVX2"
    enabled = os.environ.get("NPY_ENABLE_CPU_FEATURES")
    forced = f", NPY_ENABLE_CPU_FEATURES={enabled!r}" if enabled is not None else ""
    where = f"numpy {np.__version__} dispatches {simd}{forced}"
    if exact_dispatch():
        return f"exact tier: {where}, as v6's machine did; digests and fingerprints both checked"
    return f"tolerance tier: {where}, not AVX512_SKX, so v6's digests were not compared, only the fingerprints"


def weights(size: int) -> Iterator[np.ndarray]:
    """The PROJECTIONS weight rows for an array of ``size`` values, one at a time (the largest array is 2.6 M values).

    Row k is ``np.random.default_rng(SEED).standard_normal((PROJECTIONS, size))[k]``: drawn in sequence, the same numbers.
    """
    generator = np.random.default_rng(SEED)
    for _ in range(PROJECTIONS):
        yield generator.standard_normal(size)


def fingerprint(trajectory) -> Fingerprint:
    """The fingerprint of anything with ``t``, ``positions``, ``velocities`` and ``accelerations`` arrays."""
    arrays = [np.ascontiguousarray(getattr(trajectory, name), dtype=np.float64) for name in ARRAYS]
    flat = [array.reshape(-1) for array in arrays]
    values = np.full((len(ARRAYS), 2 + PROJECTIONS), np.nan)
    for index, array in enumerate(flat):
        if array.size:
            values[index, :2] = array[0], array[-1]
    for size in sorted({array.size for array in flat}):  # t has one size, the joint arrays another: one draw of weights each
        members = [index for index, array in enumerate(flat) if array.size == size]
        for k, row in enumerate(weights(size)):
            for index in members:
                values[index, 2 + k] = math.fsum((row * flat[index]).tolist())
    return Fingerprint(tuple(len(array) for array in arrays), tuple(array.size for array in flat), values)


@cache
def _store() -> tuple[dict[str, Fingerprint], str | None]:
    if not STORE.is_file():
        return {}, f"no fingerprint store at {STORE}"
    with np.load(STORE, allow_pickle=False) as data:
        records = {str(digest): Fingerprint(tuple(int(n) for n in lengths), tuple(int(n) for n in sizes), values.copy())
                   for digest, lengths, sizes, values in zip(data["digests"], data["lengths"], data["sizes"], data["values"],
                                                             strict=True)}
        probe = data["probe"].copy()
    # The weights are regenerated, not stored: if numpy ever changes default_rng's Gaussian stream, say so rather than
    # report every trajectory as changed.
    if not np.allclose(next(weights(probe.size)), probe, rtol=1e-14, atol=0.0):
        return records, (f"numpy {np.__version__}'s default_rng({SEED}).standard_normal stream is not the one the store was "
                         "taken with: re-take the fingerprints (python tests/_v6_fingerprint.py --write, on an AVX-512 machine)")
    return records, None


def stored() -> dict[str, Fingerprint]:
    """The stored fingerprints, by v6 digest."""
    return _store()[0]


def deviations(trajectory, expected: str) -> tuple[list[str], float, float]:
    """(problems, worst value deviation, worst projection deviation) of ``trajectory`` against v6's ``expected`` digest."""
    records, store_problem = _store()
    if store_problem:
        return [store_problem], math.nan, math.nan
    record = records.get(expected)
    if record is None:
        missing = f"no fingerprint for v6 digest {expected[:12]}: re-take them (python tests/_v6_fingerprint.py --write)"
        return [missing], math.nan, math.nan
    mine = fingerprint(trajectory)
    problems, worst_value, worst_projection = [], 0.0, 0.0
    for index, name in enumerate(ARRAYS):
        if (mine.lengths[index], mine.sizes[index]) != (record.lengths[index], record.sizes[index]):
            problems.append(f"{name}: length {mine.lengths[index]} (size {mine.sizes[index]}), v6's {record.lengths[index]} "
                            f"(size {record.sizes[index]})")
            continue
        ends, ends_v6 = mine.values[index, :2], record.values[index, :2]
        projections, projections_v6 = mine.values[index, 2:], record.values[index, 2:]
        ends_off, projections_off = np.abs(ends - ends_v6), np.abs(projections - projections_v6)
        worst_value = max(worst_value, float(ends_off.max()))
        worst_projection = max(worst_projection, float(projections_off.max()))
        # Written as "not all within" so that a NaN fails.
        if not np.all(ends_off <= VALUE_ATOL + VALUE_RTOL * np.abs(ends_v6)):
            problems.append(f"{name}: first/last value {ends.tolist()}, v6's {ends_v6.tolist()} "
                            f"(tolerance {VALUE_ATOL:g} + {VALUE_RTOL:g} * |v6|)")
        if not np.all(projections_off <= PROJECTION_ATOL):
            off = ~(projections_off <= PROJECTION_ATOL)
            k = int(np.argmax(np.where(np.isnan(projections_off), np.inf, projections_off)))
            problems.append(f"{name}: projection {k} is {float(projections[k])!r}, v6's {float(projections_v6[k])!r}, off by "
                            f"{projections_off[k]:.3g} (tolerance {PROJECTION_ATOL:g}); {int(off.sum())} of {PROJECTIONS} off")
    return problems, worst_value, worst_projection


def check(trajectory, expected: str) -> str | None:
    """None when ``trajectory`` is v6's ``expected`` (a digest in tests/test_equivalence.py) to within its fingerprint; else
    what differs."""
    problems, _, _ = deviations(trajectory, expected)
    return "; ".join(problems) or None


def values_close(mine, theirs) -> bool:
    """Two arrays of one shape whose every value agrees to the fingerprint's value tolerance (NaN never agrees)."""
    mine, theirs = np.asarray(mine, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    return mine.shape == theirs.shape and bool(np.all(np.abs(mine - theirs) <= VALUE_ATOL + VALUE_RTOL * np.abs(theirs)))


def _cases():
    import test_equivalence as equivalence  # here, not at the top: test_equivalence imports this module

    return equivalence


def write() -> int:
    """Re-take every fingerprint from v6's arrays bit for bit, or refuse."""
    if not exact_dispatch():
        print(f"REFUSED: fingerprints are only taken from bit-exact v6 arrays, and here the digests cannot hold ({tier()})")
        return 2
    equivalence = _cases()
    records: dict[str, Fingerprint] = {}
    different = []
    for label, expected, trajectory in equivalence.v6_cases():
        if equivalence.digest(trajectory) != expected:
            different.append(label)
        elif expected not in records:
            records[expected] = fingerprint(trajectory)
    missing = sorted(equivalence.v6_digests() - records.keys())
    if different or missing:
        print(f"REFUSED: not v6 bit for bit: {different}; v6 digests with no trajectory to take them from: {missing}")
        return 1
    digests = sorted(records)
    provenance = (f"taken {datetime.date.today().isoformat()} by tests/_v6_fingerprint.py --write: numpy {np.__version__}, "
                  f"Python {platform.python_version()}, {tier()}")
    temporary = STORE.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, digests=np.array(digests), provenance=np.array(provenance),
                        lengths=np.array([records[digest].lengths for digest in digests], dtype=np.int64),
                        sizes=np.array([records[digest].sizes for digest in digests], dtype=np.int64),
                        values=np.array([records[digest].values for digest in digests]),
                        probe=next(weights(PROJECTIONS)))
    temporary.replace(STORE)
    _store.cache_clear()
    print(f"wrote {STORE} ({STORE.stat().st_size} bytes): {len(digests)} v6 trajectories, {provenance}")
    return 0


def check_all() -> int:
    """Every v6 trajectory against the store: the worst deviations seen, as a fraction of each tolerance."""
    equivalence = _cases()
    worst_value, worst_projection, failed, count = 0.0, 0.0, [], 0
    for label, expected, trajectory in equivalence.v6_cases():
        problems, value, projection = deviations(trajectory, expected)
        count += 1
        bit_for_bit = equivalence.digest(trajectory) == expected
        if problems:
            failed.append(f"{label}: {'; '.join(problems)}")
        if not math.isnan(value):
            worst_value, worst_projection = max(worst_value, value), max(worst_projection, projection)
        print(f"  {'ok  ' if not problems else 'FAIL'}  {'bit for bit' if bit_for_bit else 'differs    '}  "
              f"value {value:.2e}  projection {projection:.2e}  {label}")
    print(f"{tier()}\n{count} trajectories; worst first/last value deviation {worst_value:.3g} (tolerance {VALUE_ATOL:g} + "
          f"{VALUE_RTOL:g} * |v6|); worst projection deviation {worst_projection:.3g} (tolerance {PROJECTION_ATOL:g}, "
          f"{PROJECTION_ATOL / worst_projection if worst_projection else math.inf:.3g}x)")
    for line in failed:
        print(f"FAIL {line}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="re-take tests/v6_fingerprints.npz (refuses unless bit-exact v6)")
    action.add_argument("--check", action="store_true", help="every v6 trajectory against the store; prints the worst noise")
    arguments = parser.parse_args(argv)
    return write() if arguments.write else check_all()


if __name__ == "__main__":
    raise SystemExit(main())
