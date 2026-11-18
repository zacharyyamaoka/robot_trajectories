# robot_trajectories

Joint trajectories from what you have. You hand it a robot's limits and a configuration (a purpose plan like
`plans.pid_tuning(...)`, or records like `StepTrain(...)`), and it hands back one long robot_msgs `JointTrajectory` with every
segment tied end to start, the same motion as rest-to-rest segments you can run one at a time, and the configuration as JSON
text that regenerates it sample for sample. Every sample is inside the position, velocity and acceleration limits by
construction. It is a leaf: numpy and robot_msgs at runtime, nothing else; a robot description, a reach map, kinematics or a
dynamics model enter by shape, never by import.

**v0.0.1, local only.** The recommended splice of the 2026-11-18 API round (V2's config records inside, V3's purpose plans in
front, V4's workspace on the input ladder), promoted from the round's v6 prototype with its motion unchanged, bit for bit.
Why it is shaped this way, and how to switch to one parent instead: [docs/peps/0001-api-shape.md](docs/peps/0001-api-shape.md).

## The entry ladder

What you pass decides what it can make. Each rung includes the ones before it:

1. **Joint limits** (robot_msgs `JointLimits`: position, velocity, acceleration; jerk and a named home when you have them).
   Every joint-space kind runs here, including all eight the pendulum needs this week. `Workspace.from_limits(limits)`, or
   `Workspace.from_description(description, max_acceleration=...)` (a URDF never carries an acceleration, so it is refused
   until you state one).
2. **+ Kinematics** (forward / inverse, bam_reach's `Kinematics` shape): `workspace.with_kinematics(kinematics)`.
   A **reach map** is this rung's optional cache, not an input of its own: `workspace.with_reach_map(reach_map)` keeps every
   centred kind inside the hole-free box around where it starts, and `WorkspaceSweep` tours the reachable set itself.
3. **+ Dynamics** (a model with `inverse_dynamics(q, velocity, acceleration)`, pin_utils' shape): `workspace.with_dynamics(model)`.
4. **+ Torque limits** (`max_effort` in the `JointLimits`).

`print(workspace)` shows each rung known or missing and what that costs; `generator.kinds(use_case=...)` grades every kind
available, fallback (runs, a stated way) or locked (with the call that unlocks it).

## Quick start

The call is four lines: a workspace from your limits, a generator on it, a plan, and the trajectory.

```python
from robot_trajectories import JointTrajectoryGenerator, Workspace, plans

workspace = Workspace.from_limits(limits)                      # robot_msgs JointLimits
generator = JointTrajectoryGenerator(workspace)
program = generator.generate(plans.pid_tuning(joint="can11"))  # a plan, or records: [StepTrain(joint="can11"), Hold()]
trajectory = program.trajectory                                # one JointTrajectory; program.segments: one per step
```

[examples/quick_start.py](examples/quick_start.py) is that call on the CAN 11 pendulum's limits. Run it:

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/examples/quick_start.py
```

```
30.0 s in 4 segments on can11
       0.0 -    7.0 s  step train             StepTrain(joint='can11')
       7.0 -    9.0 s  hold                   Hold(duration_s=2.0)
       9.0 -   29.0 s  chirp                  Chirp(joint='can11')
      29.0 -   30.0 s  hold                   Hold() then wait
robot_msgs.trajectory.joint_trajectory.JointTrajectory: 15011 samples, peak 32°/s of 115°/s
```

More of the API, all in [tests/scenario.py](tests/scenario.py) (the round's scenario on the 1-, 2- and 6-DOF robots):

- `plans.for_use_case("pid_tuning")` and `catalog(use_case="pid_tuning")`: the plans and the record classes that serve a purpose.
- Plans add with `+` and stay configuration until `generate`: the same plan runs on any robot whose workspace has its inputs.
- `program.steps` says, per segment, what it measures and whether to wait after it; `program.tapes()` gives each segment as a
  standalone program.
- `Program.from_json(program.to_json())` regenerates the motion identically and refuses another robot's limits;
  `Program.from_trajectory(logged)` rebuilds from a logged trajectory (its `header.meta` carries the config).
- `print(Chirp.describe())`: a record's catalog card, every field with its unit (angles shown in degrees).

## This week's tapes

Every tape of the pendulum campaign (`plans.sim_to_real_tracking`): settle, repeat floor (5x), linear chirp, velocity
plateaus, step train, speed ladder, deployment classes and the held-out classes, 5.9 min in all. Each tape is written as
`NN-<tape>.config.json` (regenerates it) and `NN-<tape>.trajectory.npz` (robot_msgs' own `save_npz`; `robot_msgs.load_npz`
reads it back), plus the week joined and an `index.json`. The limits come from a file, so the package holds no rig's values;
[examples/can11_limits.json](examples/can11_limits.json) is the CAN 11 station's.

```bash
/home/bam/robot_trajectories/.venv/bin/python -m robot_trajectories.this_week --limits /home/bam/robot_trajectories/examples/can11_limits.json --out /home/bam/robot_trajectories/outputs/this_week
```

These are the round's v6 tapes, array for array and config for config (`tests/test_equivalence.py`). What they say about
CAN 11: its 7.39 rad/s² acceleration limit caps the speed ladder at 74 % of top speed (the 80, 90 and 100 % rungs all play at
74 % and say so), and 48 of 60 deployment legs run slower than drawn; reaching top speed needs a higher acceleration limit in
the station's schedule, not a different tape.

## Adding a kind

One new file in `src/robot_trajectories/kinds/`, no other edit: `kinds/__init__.py` imports every module in its folder, and
`Kind.__init_subclass__` makes the class a frozen keyword-only record and registers it (refusing a second claim on the same
catalog row). `catalog()`, `generator.kinds()` and JSON round trips pick it up. `kinds/sine_dwell.py` is the worked example:

```python
from .. import kernels as k
from ..kind import Context, Kind, param


class SineDwell(Kind):
    """One sine at a fixed frequency on one joint, faded in and out: the gain and phase at that one frequency."""

    catalog_id = "faded_sine"   # its row in the trajectory catalog
    family = "excitation"
    use_cases = ("bandwidth",)

    joint: str | int | None = param(None, "joint", "the joint that moves (name or index; optional on a 1-DOF robot)")
    frequency_hz: float = param(1.0, "Hz", "the one frequency")
    duration_s: float = param(10.0, "s", "how long it dwells there, fades included")
    fade_s: float = param(1.0, "s", "raised-cosine fade in and out")
    margin: float = param(0.8, "1", "the tightest of room, velocity and acceleration the amplitude reaches")

    def pieces(self, context: Context) -> list[k.Arrays]:
        return [k.chirp(context.start, context.joint(self.joint), context.room_around(context.start), context.rate_hz,
                        start_hz=self.frequency_hz, end_hz=self.frequency_hz, duration_s=self.duration_s, logarithmic=False,
                        fade_s=self.fade_s, margin=self.margin, label=self.display_label)]
```

A new plan is one decorated function in `plans.py` returning `[step(Record(...), "what it measures"), ...]`. A new catalog row
is one `TrajectoryKind(...)` in `trajectory_catalog/catalog.py`; then regenerate the JSON:

```bash
/home/bam/robot_trajectories/.venv/bin/python -m robot_trajectories.trajectory_catalog.make_json
```

## What is in it

| module | what |
|---|---|
| `workspace` | `Workspace`: the robot at the rung of knowledge it was built with; `has` / `gap` / `status`, the printed ladder |
| `kind`, `kinds/` | `Kind` (a frozen config record, one file per kind), `param`, `catalog()`, `Context.room_around` |
| `generator`, `program` | `JointTrajectoryGenerator.generate(...)` -> `Program` (`.trajectory`, `.segments`, `.steps`, `.bands`, `to_json`) |
| `plan`, `plans` | `Plan` / `step`: records with a purpose and a "then wait"; the menu of purpose plans |
| `time_law`, `signals` | `TimeLaw` (a path x a time law; minimum-jerk built), `constant_speed_chirp` |
| `kernels` | the round's shared math, vendored byte for byte (min-jerk, chirp, multisine, steps, sweeps, grids, tours, join) |
| `trajectory_catalog` | the 61-row catalog (Python records, `catalog.json` generated): use cases, needs, entry level, prior art |
| `this_week` | the pendulum campaign's tapes as files (`python -m robot_trajectories.this_week`) |

`catalog()` lists the 15 record classes that generate; `trajectory_catalog` lists the 61 kinds worth supporting, 14 of which a
record claims today. Runtime dependencies: numpy and robot_msgs.

## Not built yet

- **Cartesian paths.** Every kind here is joint-space. The kinematics slot exists and gates: `with_kinematics(k)` unlocks the
  catalog rows that need FK/IK, but no record generates from it yet (the five-bar's snake paths are its first customer).
- **The dynamics slot** likewise only gates (optimised excitation, torque feasibility).
- **Time laws other than minimum-jerk** (trapezoid, S-curve via Ruckig, time-optimal via TOPP-RA) are refused by name.
- **Torque profiles, event-ended command sequences and pose sets** (17 of the catalog's 61 rows) are catalogued, not generated.
- **Collision checks** the catalog lists are reported as "not checked here", not done.

## Run the tests

Each test file is also a plain script, which is how to run them while the arm is powered (no pytest on the rig's host). The
round's ruler (G1-G4):

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_scenario.py
```

v6's 118 claims:

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_selfcheck.py
```

The catalog's rules:

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_catalog.py
```

The leaf rule:

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_boundary.py
```

v6's motion, bit for bit:

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_equivalence.py
```

With the arm off, pytest runs the same functions:

```bash
/home/bam/robot_trajectories/.venv/bin/python -m pytest /home/bam/robot_trajectories/tests -q
```

To rebuild the venv:

```bash
uv sync --project /home/bam/robot_trajectories --extra test
```
