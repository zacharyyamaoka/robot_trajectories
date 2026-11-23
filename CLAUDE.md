# robot_trajectories — working notes for agents

A signal library: every kind of trajectory is a plain function `f(space, ..., start=None) -> robot_msgs JointTrajectory`,
registered with `@kind`. The decision record is `docs/peps/0002-signal-library.md` (it supersedes 0001); read it before
changing the API. Adding kinds: `docs/KINDS_TODO.md` is the contract the lanes work to (write sets, outputs, test templates).

## Rules that are Zach's (keep the `WHY:` comments that carry them)

- **Task-agnostic, manifestation-injected** (Zach, 2026-11-18): "robot trajectories kinda exist completely independent from
  any actual physical manifestation or task". The robot enters a `JointSpace` by shape (limits, then FK/IK, dynamics, collision
  scene, self-collision); the task enters as points, poses and numbers. **No task words in a kind's name, parameters or
  labels** (no grasp, gripper, belt, conveyor, item, object, bin); what a kind serves is a use-case tag.
  `tests/kind_contract.py` checks labels.
- **Leaf rule.** Runtime: numpy + robot_msgs (editable `/home/bam/robot_msgs`) + the stdlib, nothing else. A RobotDescription,
  ReachMap, kinematics, dynamics model or collision checker enters by shape, never by import; plugins (Ruckig, TOPP-RA, a
  robot's own kinds) arrive by entry point (`robot_trajectories.kinds`, `robot_trajectories.time_parameterizations`; the old
  `robot_trajectories.time_laws` is still read), never named here.
  Never import bam_reach, bam_robot_descriptions, bam_collision, pin_utils, sim2real, robot_models or the rigs.
  `tests/test_boundary.py` enforces it.
- **No instance values in the package.** A rig's limits live in `examples/` or `tests/fixtures/`, and come in as arguments.
- **Python records are the ground truth.** A catalog row is a `TrajectoryKind` in `trajectory_catalog/catalog.py`;
  `catalog.json` is generated (`python -m robot_trajectories.trajectory_catalog.make_json`), never edited.
- **One file per kind** in `src/robot_trajectories/kinds/`, found automatically; adding one needs no other edit.
- **The vendored kernels are byte-identical to the round's** (`shared/kernels.py` at bam_ws `b5e1f6965`). Change them only
  deliberately: edit below the provenance comment, update `KERNELS_SHA256` and the digests in `tests/test_equivalence.py`.
- **v6's motion is held bit for bit** (`tests/test_equivalence.py`: the scenario, the week, the tapes, every ported kind and
  plan on three robots). A digest changes only with a deliberate change to the motion, said in the commit message.
- **Bit for bit only where numpy dispatches AVX512_SKX** (AVX2-only CPUs, e.g. CI's EPYC 7763, differ by <= 7.1e-12); every CPU
  holds the same trajectories to stored fingerprints (`tests/_v6_fingerprint.py`; after a deliberate digest change re-take
  `tests/v6_fingerprints.npz` with `--write` on an AVX-512 machine), and CI reruns `test_equivalence.py` under AVX2 dispatch.
- **SI in code; degrees on every display** (summaries, notes, `describe()`).
- Missing something robot_msgs should have? Write it here and propose the move in a PEP; never edit robot_msgs from here.

## Core files (a change to one is a barrier: land it first, alone)

`registry.py` (the `@kind` contract, `kinds()`, `catalog()`), `segments.py` (guard, chain, split, repeat, annotate), `space.py`
(JointSpace and its protocols), `time_parameterization.py` (TimeParameterization, the time-parameterization backend seam),
`cartesian.py` (IK along tool paths), `moving.py` (the moving-target contract), `plugins.py` (entry points by role), `recipe.py`
(rebuild, summary).

**Names (Zach, 2026-11-23).** The stage is *trajectory generation*; the class role is a *time parameterization* (MoveIt's name:
a path and limits in, a timed trajectory out); "time law" was an agent coinage he never used. The old names (`TimeLaw`,
`TimeLawBackend`, `register_time_law`, `time_laws`, `time_law.py`, the `time_law=` / `approach_time_law=` keywords, the
`robot_trajectories.time_laws` group) stay as silent deprecated aliases, because arm_motion_bench and its frozen datasets use
them; a recipe still records those parameters under the old keys (`registry.RECORDED_AS`). Keep both until Zach drops them.

## Tests

While the arm is powered, never pytest: run each file as a plain script, through the round's guard or pinned
(`taskset -c 2-5,16,17 nice -n 19`), one at a time. With the arm off, pytest runs the same functions.

```bash
taskset -c 2-5,16,17 nice -n 19 /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_scenario.py
```

The files: `test_scenario.py` (the round's ruler, G1-G4), `test_equivalence.py` (v6, bit for bit), `test_contract.py` (the kind
contract), `test_cartesian.py` (FK/IK passed in), `test_plugins.py` (entry points), `test_selfcheck.py` (v6's claims, carried
over), `test_catalog.py`, `test_boundary.py`, `test_time_parameterization_names.py` (the old names, keyword and entry-point
group still work). Don't edit a green test except to add. A new kind's test calls
`kind_contract.check_kind(trajectory, space)`.

## Machine and repo

- At most 2 worker processes, pinned `taskset -c 2-5,16,17`; never CPUs 22-23 (the fdcanusb IRQ). Never touch
  /dev/ttyACM*, CAN, cameras or moteus.
- **Local git only: no remote, never push.** Making it public (a GitHub repo, a package release) is Zach's call.
- Outputs (tapes, scratch) go in `outputs/` (gitignored).
