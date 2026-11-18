# robot_trajectories — working notes for agents

Joint trajectories from what you have (joint limits, then kinematics, dynamics, torque limits): config records and purpose
plans in, robot_msgs JointTrajectories inside the limits out. The decision record is `docs/peps/0001-api-shape.md`; read it
before changing the API's shape.

## Rules that are Zach's (keep the `WHY:` comments that carry them)

- **Leaf rule.** Runtime: numpy + robot_msgs (editable `/home/bam/robot_msgs`) + the stdlib, nothing else. A
  RobotDescription, ReachMap, kinematics or dynamics model enters by shape (`joint_limits()`, `joint_solutions` /
  `valid_mask`, `forward` / `inverse`, `inverse_dynamics`), never by import. Never import bam_reach, bam_robot_descriptions,
  pin_utils, sim2real, robot_models or the rigs: they will import this. `tests/test_boundary.py` enforces it.
- **No instance values in the package.** A rig's limits live in `examples/` or `tests/fixtures/`, and come in as arguments.
- **Python records are the ground truth.** A kind is a frozen `Kind` record; a catalog row is a `TrajectoryKind` in
  `trajectory_catalog/catalog.py`. `catalog.json` is generated (`python -m robot_trajectories.trajectory_catalog.make_json`),
  never edited; `tests/test_catalog.py` fails when it is stale.
- **One file per kind** in `src/robot_trajectories/kinds/`, found automatically; adding one needs no other edit.
- **The vendored kernels are byte-identical to the round's** (`shared/kernels.py` at bam_ws `b5e1f6965`). Change them only
  deliberately: edit below the provenance comment, update `KERNELS_SHA256` and the v6 digests in
  `tests/test_equivalence.py`, and say why in the commit message.
- **SI in code; degrees on every display** (summaries, notes, `describe()`).
- Missing something robot_msgs should have? Write it here and propose the move in a PEP; never edit robot_msgs from here.

## Tests

While the arm is powered, never pytest: run each file as a plain script, pinned, one at a time. With the arm off, pytest runs
the same functions (`uv run --extra test pytest -q`).

```bash
taskset -c 2-5,16,17 nice -n 19 /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_scenario.py
```

The files: `test_scenario.py` (the round's ruler, G1-G4), `test_selfcheck.py` (v6's 118 claims), `test_catalog.py`,
`test_boundary.py`, `test_equivalence.py` (v6's motion, bit for bit). Don't edit a green test except to add; a digest
changes only with a deliberate change to the motion.

## Machine and repo

- At most 2 worker processes, pinned `taskset -c 2-5,16,17`; never CPUs 22-23 (the fdcanusb IRQ). Never touch
  /dev/ttyACM*, CAN, cameras or moteus.
- **Local git only: no remote, never push.** Making it public (a GitHub repo, a package release) is Zach's call.
- Outputs (tapes, scratch) go in `outputs/` (gitignored).
