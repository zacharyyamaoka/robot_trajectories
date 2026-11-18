"""This week on the pendulum: every tape of the sim-to-real campaign (``sim_to_real_tracking``), from bare joint limits.

    /home/bam/robot_trajectories/.venv/bin/python -m robot_trajectories.this_week --limits /home/bam/robot_trajectories/examples/can11_limits.json --out /home/bam/robot_trajectories/outputs/this_week

Writes, per tape (one segment of the week), to ``--out``:
  NN-<tape>.recipe.json     the tape's recipe: robot_trajectories.rebuild(text) regenerates it sample for sample
  NN-<tape>.trajectory.npz  the trajectory, robot_msgs-native (robot_msgs.save_npz; robot_msgs.load_npz reads it back)
plus week.recipe.json / week.trajectory.npz (every tape joined) and index.json (the printed table, as data).

``--limits`` is a JSON file holding robot_msgs JointLimits as ``space.limits_to_json`` writes them, bare or under a ``"limits"``
key beside notes (``examples/can11_limits.json`` is the CAN 11 station's). WHY a file and not a constant here: the package
holds no instance values; a rig's limits live with the rig (its schedule, an example, a test fixture).

The tapes are v6's, array for array (tests/test_equivalence.py). What changed in 0.1.0: the week is the higher-order kind
``sim_to_real_tracking(space, joint)``; a tape is ``split(week)[i]``; its text is a recipe (``.recipe.json``, the call that makes
that one tape) instead of v6's program config (``.config.json``).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from robot_msgs import load_npz, save_npz
from robot_msgs.trajectory import JointLimits, JointTrajectory

from .kinds.chirp import chirp
from .kinds.sim_to_real_tracking import sim_to_real_tracking
from .recipe import rebuild, recipe
from .segments import notes, segment_info, split
from .signals import constant_speed_chirp
from .space import JointSpace, limits_from_json


def read_limits(path: str | pathlib.Path) -> tuple[JointLimits, str]:
    """(the JointLimits, a name for them) from a JSON file: bare ``limits_to_json`` output, or under a ``"limits"`` key."""
    path = pathlib.Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    values: Mapping[str, Any] = data["limits"] if "limits" in data else data
    return limits_from_json(values), str(data.get("name") or path.stem)


def week(limits: JointLimits, joint: str | None = None, campaign_round: int = 0) -> JointTrajectory:
    """The whole week as one trajectory, one segment per tape (every kind in it needs joint limits only)."""
    return sim_to_real_tracking(JointSpace.from_limits(limits), joint, campaign_round=campaign_round)


def tape_name(index: int, label: str) -> str:
    return f"{index:02d}-" + "".join(c if c.isalnum() else "-" for c in label.lower()).strip("-").replace("--", "-")


def table_row(index: int, entry: Mapping[str, Any], tape: JointTrajectory) -> dict:
    limits = tape.limits
    return {
        "tape": tape_name(index, entry["label"]), "call": entry["call"]["kind"], "purpose": entry.get("purpose", ""),
        "duration_s": round(float(tape.t[-1]), 2),
        "peak_speed_pct": round(100 * float(np.max(np.abs(tape.velocities) / limits.max_velocity)), 1),
        "peak_acceleration_pct": round(100 * float(np.max(np.abs(tape.accelerations) / limits.max_acceleration)), 1),
        "range_deg": [round(math.degrees(float(tape.positions.min())), 2), round(math.degrees(float(tape.positions.max())), 2)],
        "notes": list(entry.get("notes", ())),
    }


def chirp_before_after(limits: JointLimits, joint: str | None = None) -> list[dict]:
    """The shared chirp (one amplitude for the sweep) against the constant-speed rule, on the ruler's 0.2-5 Hz band.

    "Before" is the shared chirp's one amplitude (its swing in the unfaded middle). "After" is the swing the new rule asks for
    as the sweep passes each frequency (its envelope). WHY the envelope and not a measured swing: at 5 Hz both sweeps are
    already fading out, so a measurement there reads the fade, not the rule."""
    space = JointSpace.from_limits(limits)
    bounds = space.bounds
    index = space.joint_index(joint)
    before = chirp(space, joint, start_hz=0.2, end_hz=5.0, duration_s=20.0)
    middle = (before.t > 1.0) & (before.t < before.t[-1] - 1.0)
    shared_swing = float(np.max(np.abs(before.positions[middle, index] - space.home[index])))
    _, info = constant_speed_chirp(space.home, index, bounds, start_hz=0.2, end_hz=5.0, duration_s=20.0)
    room = float(min(space.home[index] - bounds.low[index], bounds.high[index] - space.home[index]))
    rows = []
    for at_hz in (0.2, 0.5, 1.0, 2.0, 3.0, 5.0):
        asked = float(np.interp(at_hz, info["frequency_hz"], info["envelope_rad"]))
        ceiling = min(float(bounds.max_acceleration[index]) / (2 * math.pi * at_hz) ** 2, room)  # the most any rule can swing here
        rows.append({"hz": at_hz, "before_deg": round(math.degrees(shared_swing), 2), "after_deg": round(math.degrees(asked), 2),
                     "ceiling_deg": round(math.degrees(ceiling), 2)})
    return rows


def write_tapes(limits: JointLimits, out: str | pathlib.Path, *, joint: str | None = None, campaign_round: int = 0,
                robot: str = "") -> dict:
    """Generate the week, write every tape and its recipe, check each reads back and regenerates; return index.json's data."""
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    whole = week(limits, joint, campaign_round)
    rows = []
    for index, (entry, tape) in enumerate(zip(segment_info(whole), split(whole), strict=True), start=1):
        row = table_row(index, entry, tape)
        (out / f"{row['tape']}.recipe.json").write_text(recipe(tape) + "\n")
        path = out / f"{row['tape']}.trajectory.npz"
        save_npz(tape, path)
        back = load_npz(path)
        if not (np.array_equal(back.positions, tape.positions) and np.array_equal(back.t, tape.t)):
            raise SystemExit(f"{path} does not read back identically")
        again = rebuild((out / f"{row['tape']}.recipe.json").read_text())
        if not np.array_equal(again.positions, tape.positions):
            raise SystemExit(f"{row['tape']}.recipe.json does not regenerate its trajectory")
        rows.append(row)
    (out / "week.recipe.json").write_text(recipe(whole) + "\n")
    save_npz(whole, out / "week.trajectory.npz")
    call = json.loads(recipe(whole))["calls"][0]
    index = {"call": call, "robot": robot or "bare JointLimits", "tapes": rows, "chirp_before_after": chirp_before_after(limits, joint),
             "notes": notes(whole), "duration_s": round(float(whole.t[-1]), 2)}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


def _limits_text(limits: JointLimits) -> str:
    low, high = np.degrees(limits.min_position), np.degrees(limits.max_position)
    span = "±" + f"{high[0]:.1f}°" if len(low) == 1 and math.isclose(low[0], -high[0]) else f"{np.round(low, 1).tolist()}..{np.round(high, 1).tolist()}°"
    velocity = ", ".join(f"{value:.0f}" for value in np.degrees(limits.max_velocity))
    acceleration = ", ".join(f"{value:.0f}" for value in np.degrees(limits.max_acceleration))
    return f"{span}, {velocity}°/s, {acceleration}°/s²"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robot_trajectories.this_week", description=__doc__.split("\n\n")[0])
    parser.add_argument("--limits", required=True, help="JSON file with the robot's JointLimits (examples/can11_limits.json)")
    parser.add_argument("--out", required=True, help="directory to write the tapes to (created if missing)")
    parser.add_argument("--joint", default=None, help="the joint that moves (default: the only joint of a 1-DOF robot)")
    parser.add_argument("--round", type=int, default=0, help="the campaign round: it seeds the speed ladder's held-out pairs")
    args = parser.parse_args(argv)
    limits, name = read_limits(args.limits)
    joint = args.joint
    if joint is None and len(limits.joint_names) == 1:
        joint = limits.joint_names[0]  # WHY named, not None: the tapes' configs then say which joint, as v6's did
    index = write_tapes(limits, args.out, joint=joint, campaign_round=args.round, robot=name)
    rows = index["tapes"]
    limit_words = _limits_text(limits)
    call = index["call"]
    print(f"This week on {name}: sim_to_real_tracking(space, joint={call['joint']!r}), {len(rows)} tapes, "
          f"{index['duration_s'] / 60:.1f} min in all, from bare "
          f"JointLimits ({limit_words})")
    print(f"  {'tape':<34} {'duration':>9} {'peak speed':>11} {'peak accel':>11}   swept range")
    for row in rows:
        low, high = row["range_deg"]
        print(f"  {row['tape']:<34} {row['duration_s']:>7.1f} s {row['peak_speed_pct']:>9.1f} % {row['peak_acceleration_pct']:>9.1f} %"
              f"   {low:+6.1f}° .. {high:+5.1f}°  ({high - low:.1f}°)")
    print("  peak speed / accel: the tape's worst sample as a % of the limit")
    print("\nWhat the kinds reported while generating:")
    for note in index["notes"]:
        print(f"  - {note}")
    print("\nChirp swing, 0.2-5 Hz over 20 s (amplitude, ± degrees):")
    print(f"  {'frequency':>9}  {'shared kernels.chirp':>21}  {'constant-speed rule':>20}  {'ceiling (accel, room)':>22}")
    for row in index["chirp_before_after"]:
        print(f"  {row['hz']:>6.1f} Hz  {row['before_deg']:>19.2f}°  {row['after_deg']:>18.2f}°  {row['ceiling_deg']:>20.2f}°")
    print("  shared: one gain for the whole sweep, set by its top end; rule: v / 2 pi f, capped by 0.85 x the room and")
    print("  0.85 x a / w^2. The ceiling is the most any amplitude rule can swing at that frequency.")
    print(f"\nWrote {2 * len(rows) + 3} files to {pathlib.Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
