# PEP 0001: the API shape — the recommended splice of the 2026-11-18 round

- **Status:** Accepted as the AI default 2026-11-18, pending Zach's pick. Taken under his standing rule for reversible work
  ("your recommendation is the default, take it"), so switching later is cheap: see "Switching to one parent" below. Undo:
  the repo is local and private; `git revert` any commit, or delete `/home/bam/robot_trajectories`.
- **Ask:** Zach, Daily Note 2026-11-18: "You pass in a robot description and a configuration, and then you get out like a
  very long joint trajectory that kinda like the start and end of each segment ties together... and you can of course choose
  to have them returned separately so you can execute one, then wait." Plus a catalog of every kind of trajectory, and entry
  points from a RobotDescription, a ReachMap or just joint limits.
- **The round:** five API proposals, one frozen scenario check, two judges:
  [bam_ws docs/trajectory_generator](https://github.com/zacharyyamaoka/bam_ws/tree/main/docs/trajectory_generator)
  (`origin/main` @ `b5e1f6965`). Design note: vault `DP - Joint Trajectory Catalog Generator` ("Refined spec",
  "The splice, built on the recommendation (v6)").
- **This package** is that round's v6 prototype (`variants/v6_splice/`) promoted to a real leaf package, its motion
  unchanged bit for bit (`tests/test_equivalence.py`).

## 1. The answer in six lines

1. **A kind of trajectory is a frozen, keyword-only config record** (V2): `Chirp(joint="can11", start_hz=0.2)` is the config,
   the repr, the JSON and the catalog card at once. One file per kind, found automatically.
2. **`generator.generate(...)` returns a `Program`** (V2): `.trajectory` (one JointTrajectory, segments tied end to start),
   `.segments` (one rest-to-rest JointTrajectory per record), `.to_json()` / `Program.from_json(text)` (identical, and a
   different robot is refused).
3. **Purpose plans are the front door, and robot-free** (V3): `plans.pid_tuning(joint="can11")` is a `Plan` of steps, each a
   record with what it measures and whether to wait after it; plans add with `+`. The robot enters only at `generate`.
4. **The robot is a `Workspace` on Zach's input ladder** (V4): joint limits -> + kinematics -> + dynamics -> + torque limits,
   a reach map being the kinematics rung's optional cache. It prints each rung known or missing, and grades every kind
   available / fallback / locked with the call that unlocks it.
5. **A path kind takes a `TimeLaw`** (new, from the refined spec): the speed ladder is one A-to-B path under six time laws.
6. **The math is the round's shared kernels, vendored byte for byte**, so every proposal's results still hold here.

## 2. The round: five proposals, frozen criteria, the scores

| | Name | The thing you hold | Where the robot enters |
|---|---|---|---|
| V1 | Signal library | a function per kind (`chirp(space, ...)`), scipy.signal-style | a `JointSpace` value |
| V2 | Config records | a frozen dataclass per kind, Keras/Hydra-style; `generate([...]) -> Program` | the generator's constructor |
| V3 | Use-case plans | a plan named for its purpose (`plans.pid_tuning(...)`), steps with a "why" | duck-typed, first argument |
| V4 | Staged pipeline | a generator bound to a workspace, a method per kind, each gated on what it needs | staged and gated |
| V5 | Trajectory algebra | a lazy shape expression (`steps(5) >> hold(2) >> chirp(0.2, 5)`) | bound last |

The criteria, weighted before any variant was built (sum 100): **fr1** from what you have to a trajectory you can run, in one
call (25); **fr2** one long trajectory, or the same as segments (20); **fr3** the catalog browsable by use case and robot (20);
**fr4** the landing page says intent and API at a glance (20); **fr5** adding a kind is small and local (10); **fr6**
reproducible and loggable (5). Hard gates (pass/fail, never averaged): G1 robot_msgs out, G2 the leaf rule, G3 safe by
construction, G4 the scenario. All five passed every gate.

Two judges scored each criterion 1-5: a **rig user** (the bench: "what do I run to tune this PID?") and a **maintainer**
(whoever grows it to 61 kinds).

| | combined | rig user | maintainer |
|---|---|---|---|
| **V2** Config records | **84.0** | 85.0 | 83.0 |
| **V3** Use-case plans | **82.8** | 93.5 | 72.0 |
| **V4** Staged pipeline | **82.0** | 81.5 | 82.5 |
| V1 Signal library | 81.2 | 84.0 | 78.5 |
| V5 Trajectory algebra | 79.0 | 83.5 | 74.5 |

**The hinge: whose view you weigh.** V2 leads, but only just; it is a three-way tie, not a winner. At the bench V3 is far
ahead (93.5 against 85.0): "what do you want to test?" is the question actually asked, and its steps say what they measure
and when to wait. For the maintainer V3 drops to last (72.0): each plan is hand-kept, while V2 (83.0) and V4 (82.5) keep each
kind's metadata with its code, and V4's stage gating already matched the FK/IK ladder of the refined spec. Move 10 points from
fr5 onto fr4 and V3 leads.

## 3. Why the splice, and from which parent

V2 is the only proposal both judges rated high on everything, so it is the core; the other two leaders each add the thing
they won on, made to fit it:

| Part | From | Why this parent |
|---|---|---|
| A kind is a frozen keyword-only record class, ClassVar tags, auto-registered, one file per kind (`kind.py`, `kinds/`) | **V2** | The line you write is the config, the repr, the JSON and the catalog card. fr5 is one file. |
| `generate(...)` -> `Program` (`.trajectory`, `.segments`, `to_json` / `from_json` / `from_trajectory`, fingerprint refusal) | **V2** | One segment per record keeps `zip(records, segments)` aligned; the config holds every field and the limits used, so a log rebuilds. |
| `plans.pid_tuning(...)` etc., a `Plan` of steps with `purpose` and `then`, plans add with `+` (`plan.py`, `plans.py`) | **V3**, made robot-free | At the bench V3 led: the question you ask is the purpose. Robot-free because a plan is configuration: V3 needed the robot first, so a plan could not be written before the robot was chosen. |
| `Workspace` with `has` / `gap` / `status`, printed rungs, available / fallback / locked with the fix (`workspace.py`, `generator.kinds()`) | **V4**, on Zach's ladder | V4's stage gating already matched the refined spec; the stages are now joint limits -> kinematics -> dynamics -> torque limits. |
| `room_around(start)`: the hole-free box every centred kind uses | all five | Every proposal converged on it independently; a chirp on the five-bar stays reachable without knowing what a reach map is. |
| `TimeLaw`, `LinearChirp`'s constant-speed amplitude, the eight "now" kinds, `Repeat` | new | The refined spec and this week's pendulum campaign. |
| Not taken: V1's free functions, V5's algebra (`>>`, `+`, `.repeat`) | V1, V5 | V5's `.repeat(n)` survives as the `Repeat` record; V1's `JointSpace.from_*` became the workspace's constructors. |

## 4. The refined spec (Zach's further thoughts, 2026-11-18)

1. **The entry ladder is joint limits -> + kinematics -> + dynamics (-> + torque limits); a reach map is a cache, not an
   input.** Joint limits cover every joint-space kind, including all eight "now" kinds. Kinematics is duck-typed on bam_reach's
   `Kinematics` protocol (`forward`, `inverse`, `joint_names`, `cache_key`), never imported.
2. **Cartesian limits are two things, and only one is new:** how fast the tool may move (robot_msgs `CartesianLimits` already
   has it) and where it can be (the image of the joint box under FK, sampled and cached: a reach map built on demand).
3. **Paths vs trajectories: split on what a kind is made of.** A *time signal* (chirp, multisine, step train, holds, torque
   profiles) is its own time law; a *path* (waypoints, grid snakes, the speed ladder) is geometry plus a separate time law. So
   `Path x time law -> JointTrajectory` and `Signal -> JointTrajectory` directly; the catalog's `starts_as` says which.
4. **At a glance: 1D / 2D / 3D / 6D** — the catalog carries `dimensions` and an `entry_level` derived from its needs.
5. **Push-back kept: the first release needs no Cartesian path.** Every "now" kind runs on joint limits; the kinematics slot
   exists, gates and generates nothing yet, so the API does not change shape when the five-bar's snake paths move in.

## 5. Decisions inside the splice (from v6)

1. **The robot enters once, at `generate`, as a workspace.** Plans and records never hold a robot, so the same
   `plans.sim_to_real_tracking(joint=...)` runs on the pendulum, the five-bar and the UR5e, and a plan can be saved and reviewed
   before anyone picks a robot.
2. **The kinematics and dynamics slots exist, gate, and generate nothing yet.** On the UR5e, filling them unlocks catalog rows
   30 -> 38 -> 41 of 61 (`tests/test_selfcheck.py` §8). For `WorkspaceSweep`, kinematics alone is still a fallback: sampling
   the reachable set from it is not built.
3. **A path kind takes a `TimeLaw`; only minimum-jerk is built.** Trapezoid, S-curve and TOPP-RA are refused by name. The
   ruler's `RandomWaypoints` / `WorkspaceSweep` keep the kernels' `speed` so the round's scenario stays bit-identical; moving
   them onto `TimeLaw` is a one-field change.
4. **The constant-speed chirp is a new function; the ruler's chirp keeps the shared kernel.** On CAN 11 it swings ±16.1° at
   0.5 Hz where the shared chirp holds ±0.47° across the whole sweep; at 5 Hz the acceleration limit itself allows ±0.43°.
5. **Faithful to the code the catalog cites, inside the limits.** Where the rig's code runs past a limit (PID tuna's 20 ms
   step, the rack's stepped friction plateaus), the kind keeps the shape and the draws and makes the timing feasible, and the
   program notes say where.

## 6. What changed when v6 became this package

- **The math is vendored.** `robot_trajectories/kernels.py` is the round's `shared/kernels.py` at `b5e1f6965` byte for byte
  below a provenance comment (sha256 held by `tests/test_equivalence.py`); the package no longer reaches into bam_ws.
- **The catalog moved in** as `robot_trajectories.trajectory_catalog` (`kind.py` vocabulary, `catalog.py` records,
  `make_json.py`, generated `catalog.json`). Not named `catalog`: a submodule of that name would shadow the `catalog()`
  function. `kind.py`'s entry-ladder constants now come from the catalog's vocabulary instead of a copy; the round's
  `mini_catalog` use-case tags became literal tuples in the six kinds that read them (same values, same order).
- **`this_week` is a module with a CLI** (`python -m robot_trajectories.this_week --limits FILE --out DIR`). The limits come
  from a file (`examples/can11_limits.json`), so the package holds no rig's values.
- **Tests:** the round's ruler (G1-G4), v6's 118 claims, the catalog's rules, the leaf rule, and v6 equivalence, each also a
  plain script for the powered rig's host.
- **Motion: none.** The scenario's three sessions, this week's eight tapes, their configs and index rows equal v6's.

## 7. Switching to one parent

Each parent's part is its own module, so a pick is mostly deletions. Nothing in `kinds/` changes in any of the three: the
records and their math are the shared core.

- **V2 alone:** delete `plan.py` and `plans.py`; `generate()` keeps accepting a list of records (`_flatten` loses two
  branches); `Program.steps` becomes `Program.records` (the JSON's `steps` lose `purpose` / `then` / `plan`). Optionally fold
  `Workspace` back into V2's `from_limits / from_reach_map / from_description` constructors (they already exist as shorthands
  on `JointTrajectoryGenerator`). Kinds, TimeLaw, signals, Repeat: unchanged.
- **V3 alone:** keep `plans.py` as the front door and make plans eager again (`plans.pid_tuning(robot, ...)` returns the
  generated Program instead of a Plan). Records stay underneath, as V3's kinds were; nothing else moves.
- **V4 alone:** keep `workspace.py` and `generator.py`; add V4's `__getattr__` dispatch (`generator.chirp(...)`,
  `with generator.session() as s: s.step_train(...)`) that builds the record from the call's keywords and appends it; delete
  `plan.py` / `plans.py`. The recipe is `program.to_json()` already.

## 8. Migration (who moves onto this, later)

From v6's README, unchanged in substance: sim2real's `schedule/joint.py` families become `Kind` subclasses (decorator keywords
become ClassVars, `make(description, rng, stretch)` becomes `pieces(context)`, a `None` becomes a declared need);
`bam_rack_twin/schedule.py`'s families map onto `LinearChirp`, `Chirp`, `VelocityPlateaus`, `StepTrain`, `Repeat` and
`DeploymentMotionClasses`; the pendulum campaign's ladder geometry and seeds are `SpeedLadder`'s; `traj_integration_tests`'
`Segment` becomes `program.segments`. `bam_traj_gen` stays the pick planner. None of that is done here.

## 9. Assumed, check before a real run

- The station's floor tape is not in the tree; the week's floor tape is `Move(targets_fraction=(0.7, -0.7),
  time_law=TimeLaw.minimum_jerk(peak_speed_fraction=0.6), dwell_s=1.0)`, five times.
- The speed ladder's held-out pairs use the campaign's rule (`round x 1000 + rung x 10 + k`) but not its seed string, and the
  usable window is the workspace's room, not the station facts' window and gravity band.
- The kinematics and dynamics slots were exercised with stand-ins of the protocols' shape, never with bam_reach or pin_utils
  objects; collision checks the catalog lists are not done.
