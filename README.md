# robot_trajectories

A signal library for robot motion. Every kind of trajectory is a plain function: you pass a `JointSpace` (the robot) and
numbers, points or poses (the task), and get back one robot_msgs `JointTrajectory`, inside the limits, at rest at both ends,
carrying the call that made it. Trajectories concatenate (`chain`), cut back into rest-to-rest pieces (`split`) and rebuild
from their own recipe, sample for sample. It is a leaf: numpy and robot_msgs at runtime, nothing else.

**Landing page: <https://zacharyyamaoka.github.io/robot_trajectories/>**. It has the quick start run for real, the API in one
table, and a catalog card for every kind with a plot of its joint positions (in degrees), built by `website/build_site.py`.

**v0.1.0, public on GitHub, not on PyPI yet.** Zach picked the round's V1 signal library on 2026-11-18 ("Its pure its functional it feels
reusable... the type is the API"); this is that, with v6's motion kept bit for bit. Decision record:
[docs/peps/0002-signal-library.md](docs/peps/0002-signal-library.md).

## The rule: task-agnostic, manifestation-injected

Zach (Daily Note 2026-11-18): "robot trajectories kinda exist completely independent from any actual physical manifestation
or task. The actual physical manifestation eventually you can inject ... via like ik/fk dynamics model collision model, etc.
and the tasks you can pass in like points and stuff. but at this level we never really know about like what the semantic
meaning is of things. that is why it works for like any robot type, or any task."

- **The robot is injected into a `JointSpace`**, never imported: joint limits, then kinematics (FK/IK), dynamics, a collision
  scene and a self-collision model, each a duck-typed protocol.
- **The task arrives as points, poses and numbers.** No kind knows what the motion is for: no task words in a kind's name,
  parameters or labels. What a kind serves is a use-case tag.
- **Every kind returns a robot_msgs JointTrajectory**, the type everything here joins on.

## Names

Zach (2026-11-23): "All TrajectoryGenerators output Trajectories... so yes there are many possible like type of generators but
they all share that commonality". The stage is still *trajectory generation*; three ideas name what happens in it:

- **`TrajectoryGenerator`**: anything that hands you back a robot_msgs `Trajectory` when you call it. Every kind here is one
  (`joint_waypoints`, `chirp`, `pick_and_place`, `speed_ladder`, ...). It is a `typing.Protocol`; `isinstance` only checks
  that the thing is callable, so the return type is the real contract.
- **`WaypointTrajectoryGenerator`**: points and joint limits in, a `JointTrajectory` out. Minimum-jerk and trapezoid are
  built in; Ruckig, Ruckig Pro and TOPP-RA come as plugins. A few points with the curve between left to the backend (Ruckig)
  and a dense path followed exactly (TOPP-RA) are the same type: following the path exactly is what a backend does, not a
  separate class.
- **A kind hands its legs to one.** A path kind says where the joints go, then hands each leg's timing to the
  `WaypointTrajectoryGenerator` you pass as `waypoint_generator=` (`approach_waypoint_generator=` for an approach leg).

Formerly TimeLaw, then TimeParameterization; old names still import (as the same objects, silently), the `time_law=` and
`time_parameterization=` keywords still work, and recipes keep their `"time_law"` keys.

## Quick start

```python
from robot_trajectories import JointSpace, chain, hold, pid_tuning, split, step_train, summary

space = JointSpace.from_limits(limits)                      # robot_msgs JointLimits: positions, velocities, accelerations
trajectory = step_train(space, joint="can11")               # a kind: f(space, ..., start=None) -> JointTrajectory
session = chain(pid_tuning(space, joint="can11"), step_train(space, joint="can11", steps_fraction=(0.3, -0.3)), hold(space, 2.0))
split(session)                                              # the same motion as rest-to-rest pieces: run one, then wait
```

[examples/quick_start.py](examples/quick_start.py) is that on the CAN 11 pendulum's limits:

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/examples/quick_start.py
```

```
35.4 s in 6 segments on can11
       0.0 -    7.0 s  step train                 step_train(space, joint='can11')
       7.0 -    9.0 s  hold                       hold(space, duration_s=2.0)
       9.0 -   29.0 s  chirp                      chirp(space, joint='can11')
      29.0 -   30.0 s  hold                       hold(space)  then wait
      30.0 -   33.4 s  step train                 step_train(space, joint='can11', steps_fraction=(0.3, -0.3))
      33.4 -   35.4 s  hold                       hold(space, duration_s=2.0)
6 rest-to-rest pieces; rebuilt from its 933-character recipe: identical
```

## The robot: a JointSpace, one rung at a time

```python
space = JointSpace.from_limits(limits)                                   # joint limits: every joint-space kind runs here
space = JointSpace.from_description(ur5e, max_acceleration=6.0).within(box)   # a URDF has no acceleration: refused until stated
space = space.with_kinematics(fk_ik)        # forward(q) -> 4x4, inverse(pose, seed) -> q | None (bam_reach's shape)
space = space.with_reach_map(reach_map)     # an optional cache: every centred kind stays in the hole-free box (space.room)
space = space.with_dynamics(model)          # torque(q, qd, qdd); forward(q, qd, tau) -> qdd for effort-commanded kinds
space = space.with_collision(scene)         # in_collision(q) -> bool: the guard then refuses any colliding sample
print(space)                                # each rung known or missing, angles in degrees
```

A kind that needs a rung the space lacks raises `MissingRung`, naming the rung and the call that adds it.

## Higher-order kinds

A kind can chain other kinds; it is then recorded as its own one call, and its segments keep the calls that make each alone.

```python
pid_tuning(space, joint="can11")            # step train, settle, chirp, settle (each segment says what it measures)
sim_to_real_tracking(space, joint="can11")  # this week's eight tapes on the pendulum, one segment each
friction_id(space, "can11"); tracking_check(space); workspace_survey(space); dynamics_excitation(space)
repeat_floor_tape(space, "can11")           # one tape five times: the run-to-run floor; repeat(trajectory, times) for any tape
```

Writing one is a function that threads `start=` from one call to the next and returns `chain(...)`
(`src/robot_trajectories/kinds/pid_tuning.py`).

## Cartesian paths, with FK/IK passed in

```python
space = JointSpace.from_description(ur5e, max_acceleration=6.0).within(box).with_kinematics(fk_ik)
cartesian_path(space, [(0.4, 0.1, 0.3), (0.4, 0.2, 0.3)])   # straight tool lines, IK every 1 mm, refused at the first failing sample
cartesian_waypoints(space, points)                          # IK at the points only, joint legs between, a dwell at each
pick_and_place(space, pick, place, lift_m=0.05)             # the U: approach, down, dwell, up, over, down, dwell, up
```

[examples/ur5e_pick_and_place.py](examples/ur5e_pick_and_place.py) runs the U on a UR5e (its stand-in kinematics are in
`tests/fixtures/robots.py`):

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/examples/ur5e_pick_and_place.py
```

Points are in metres in the kinematics' base frame; `orientation="hold"` keeps the tool's start orientation, `"follow"` takes
each pose's own. An unreachable point raises `Unreachable` naming the sample, the tool point and why. Forward kinematics of
every output lands within micrometres of the commanded tool points (`tests/test_cartesian.py` holds 1 mm on the UR5e and the
five-bar).

## Multi-waypoint trajectories

```python
joint_waypoints(space, [[0.2, -1.4, 1.2, -1.4, -1.6, 0.0], [...]],
                waypoint_generator=WaypointTrajectoryGenerator.trapezoid(peak_speed_fraction=0.8))
move(space, joint="can11", targets_fraction=(0.5, -0.5))    # waypoints as fractions of the room around the start
cartesian_waypoints(space, [p1, p2, p3], dwell_s=0.3)        # tool points
```

Trajectory generation here is a path and its timing: a path kind takes a `WaypointTrajectoryGenerator` (points and limits in,
a timed trajectory out; MoveIt calls the role a time parameterization):
`WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.6)` or `move_s=`, `WaypointTrajectoryGenerator.trapezoid(...)`,
or a plugin's (`WaypointTrajectoryGenerator("ruckig", ...)`). A leg may start or end moving (`start_state=`, `end_state=`: the
quintic through both states), which is what a velocity-matched sync needs.

## Segments and recipes

A trajectory carries its segments as a per-sample `segment` channel plus labels in `header.meta`, so slicing, resampling and
logging keep them. `split`, `chain`, `labels`, `bands` (the kernel pieces), `notes` (what the kinds reported), `annotate`
(purpose, wait after) and `repeat` are pure functions of trajectories. `recipe(t)` is the JSON of every call with every
argument; `rebuild(recipe(t))` is `t` again, and a different space is refused by fingerprint. `summary(t)` prints it.

## The catalog

`catalog(use_case="pid_tuning", space=space)` lists the kinds that generate (available, fallback or locked on that space) and,
honestly marked, the catalog rows nothing implements yet: 25 kinds generate; 50 of the 68 catalog rows are still to build,
split into six lanes in [docs/KINDS_TODO.md](docs/KINDS_TODO.md). `kinds()` is the registry. The rows themselves (use cases,
needs, sources, prior art) are `robot_trajectories.trajectory_catalog`, Python records; `catalog.json` is generated:

```bash
/home/bam/robot_trajectories/.venv/bin/python -m robot_trajectories.trajectory_catalog.make_json
```

## Adding a kind

One new file in `src/robot_trajectories/kinds/`, no other edit:

```python
from .. import kernels as k
from ..registry import kind


@kind(use_cases=("bandwidth",), catalog_id="faded_sine")
def sine_dwell(space, joint=None, *, frequency_hz=1.0, duration_s=10.0, fade_s=1.0, margin=0.8, label="sine dwell", start=None):
    """One sine at a fixed frequency on one joint, faded in and out: the gain and phase at that one frequency."""
    return k.chirp(start, space.joint_index(joint), space.room(start), space.rate_hz, start_hz=frequency_hz, end_hz=frequency_hz,
                   duration_s=duration_s, logarithmic=False, fade_s=fade_s, margin=margin, label=label)
```

The decorator refuses what the space lacks, guards the output (limits, jumps, rest, collisions), and records the call. The
contract is PEP 0002 §4; `tests/kind_contract.py`'s `check_kind(trajectory, space)` holds a new kind to it.

## Plugins: kinds and waypoint trajectory generators from other packages

Simple kinds live here; anything that needs another library (Ruckig, TOPP-RA, a robot's own stack) is a separate package that
registers by entry point, cut by role:

```toml
[project.entry-points."robot_trajectories.kinds"]
my_kinds = "my_package.kinds"

[project.entry-points."robot_trajectories.waypoint_trajectory_generators"]
ruckig = "robot_trajectories_ruckig:RuckigBackend"
```

A plugin kind is listed, tagged and guarded like a built-in one; a waypoint trajectory generator's backend serves
`WaypointTrajectoryGenerator(name, ...)`. The group's older names, `robot_trajectories.time_parameterizations` and
`robot_trajectories.time_laws`, are still read (a name declared in more than one is the newest group's). Two providers of one
waypoint trajectory generator are refused (Ruckig Pro is a drop-in for `ruckig`, never
installed beside Community), and a Ruckig backend never hands intermediate waypoints to Community (it sends them to Ruckig's
cloud). `robot_trajectories.planners` and `.optimizers` are reserved for the later roles.

## Moving targets

`robot_trajectories.moving` is the contract synchronised kinds accept (lane F builds them): a `MovingTarget` (`target(t)`,
`velocity(t)`, `stamp`, `latency_s`, `window`, `assumes`), `LinearMotion` for constant velocity (it absorbs bam_traj_gen's
`sync_dir` / `sync_speed`), `TargetWaypoint` (chase, inherit or fixed), `CannotCatch` refusals, and `synchronised(...)`, the
ruler: within tolerance over the sync window, velocity matched at the sync point. A precomputed sync is a prediction under its
`assumes`; execution replans or offsets live.

## This week's tapes

Every tape of the pendulum campaign (`sim_to_real_tracking`): settle, repeat floor (5x), linear chirp, velocity plateaus, step
train, speed ladder, deployment classes and the held-out classes, 5.9 min in all, each written as `NN-<tape>.recipe.json`
(rebuilds it) and `NN-<tape>.trajectory.npz` (robot_msgs' `save_npz`), plus the week joined and an `index.json`. The limits come
from a file; [examples/can11_limits.json](examples/can11_limits.json) is the CAN 11 station's.

```bash
/home/bam/robot_trajectories/.venv/bin/python -m robot_trajectories.this_week --limits /home/bam/robot_trajectories/examples/can11_limits.json --out /home/bam/robot_trajectories/outputs/this_week
```

The tapes are v6's, array for array (`tests/test_equivalence.py`). On CAN 11 the 7.39 rad/s² acceleration limit caps the speed
ladder at 74 % of top speed, and 48 of 60 deployment legs run slower than drawn.

## Run the tests

Each test file is a plain script (no pytest on the rig's host while the arm is powered). The round's ruler (G1-G4):

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_scenario.py
```

v6's motion, bit for bit:

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_equivalence.py
```

The kind contract (decorator, guard, recipe, segments, moving boundaries, waypoint-trajectory-generator seam, injected slots,
moving targets):

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_contract.py
```

Cartesian and multi-waypoint kinds on the UR5e and the five-bar:

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_cartesian.py
```

Plugins by entry point:

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_plugins.py
```

The renames of 2026-11-23 (the `TimeLaw` and `TimeParameterization` names, their keywords and their entry-point groups still
work; `TrajectoryGenerator` holds what the kinds return):

```bash
/home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_names.py
```

v6's claims, carried over:

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

With the arm off, pytest runs the same functions:

```bash
/home/bam/robot_trajectories/.venv/bin/python -m pytest /home/bam/robot_trajectories/tests -q
```

To rebuild the venv:

```bash
uv sync --project /home/bam/robot_trajectories --extra test
```
