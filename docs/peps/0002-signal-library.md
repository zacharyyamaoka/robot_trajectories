# PEP 0002: the signal library — plain functions in, robot_msgs JointTrajectories out

- **Status:** Accepted 2026-11-18 (Zach's pick). Supersedes [PEP 0001](0001-api-shape.md) (the V2/V3/V4 splice).
- **Package:** `robot_trajectories` 0.1.0, local at `/home/bam/robot_trajectories` (no remote). Undo: `git revert` the commit
  that lands this; 0.0.1 (the splice) is `main` @ `138fc43`.
- **Pick:** Zach, Daily Note 2026-11-18, "Feedback on Joint Trajectory Library": "I really like the Signal library approach...
  Its pure its functional it feels reusable... all of them essentially outputting joint trajectories... as its a common
  interface you can concatenate them all together... feels easier to extend.. because the type is the API... now we can make
  like high order functions as well. PID tuning trajectory is now just a function which ensembles the lower level
  functions... even the U shaped pick and place shape is also just a robot trajectory... its just that now we are like putting
  in cartesian points... thoose yes require kinda like FK/IK.. that you need to pass in. but the output is the same.. just
  joint trajectories... What about like multi waypoint trajectories?"
- **Starting point:** the round's V1 prototype (bam_ws `docs/trajectory_generator/variants/v1_signal_library/`, its
  "Decisions" and "Keep parts"), with v6's motion kept bit for bit.

## 0. The rule every kind obeys: task-agnostic, manifestation-injected

Zach, Daily Note 2026-11-18, "Robot Trajectories": "the robot trajectories kinda exist completely independent from any actual
physical manifestation or task. The actual physical manifestation eventually you can inject in that information via like
ik/fk dynamics model collision model, etc. and the tasks you can pass in like points and stuff. but at this level we never
really know about like what the semantic meaning is of things. that is important that is why it works for like any robot type,
or any task." And: "Joint Trajectories is like this unifying data type that all robot trajectories need to end up in."

So:
- **The robot is injected, never imported.** A `JointSpace` holds joint limits and, optionally, kinematics, dynamics, a
  collision scene, a self-collision model and a reach cache, each duck-typed (§5). Nothing in the package names a robot.
- **The task arrives as points, poses and numbers.** No kind knows what the motion is for: **no task words in a kind's name,
  parameters or labels** (no grasp, gripper, belt, conveyor, item, object, bin). What a kind serves is a use-case tag
  (`conveyor_pick_and_place` is a tag, not a kind). `tests/kind_contract.py` checks labels for task words.
- **Every kind returns a robot_msgs `JointTrajectory`**: the common interface that concatenates.

Excluded from the core, by Zach's call: **crab-claw offsets** ("that's not core library": a tool offset belongs to the
kinematics the caller injects) and **multi-DOF hands** ("will see"). One name predates the rule and is kept for v6 parity:
the `pickplace` motion class of `deployment_motion_classes` (PID tuna's class name, data, not a kind). `pick_and_place` is the
brief's name for the U shape; its parameters are two points and a lift, and its labels are geometric ("down to the first
point", "dwell at the second point").

## 1. The API in fifteen lines

```python
space = JointSpace.from_limits(limits)                              # or .from_description(d, max_acceleration=...), .within(box)
space = space.with_kinematics(k).with_dynamics(d).with_collision(scene).with_self_collision(m).with_reach_map(r)
trajectory = step_train(space, joint="can11")                       # every kind: f(space, ..., start=None) -> JointTrajectory
session = chain(pid_tuning(space, "can11"), friction_id(space, "can11"))   # higher-order kinds; chain joins, segments kept
split(session); labels(session); bands(session); notes(session)     # segments are a view (channels + header.meta)
repeat(tape, 5); annotate(t, "what it measures", wait_after=True)   # pure functions on trajectories
rebuild(recipe(session)) == session                                 # every call recorded, defaults written out; refuses another space
cartesian_path(space, [p1, p2]); cartesian_waypoints(space, points); pick_and_place(space, pick, place, lift_m=0.05)
joint_waypoints(space, [[...], [...]], time_law=TimeLaw.trapezoid(peak_speed_fraction=0.8))
TimeLaw.minimum_jerk(peak_speed_fraction=0.6); TimeLaw("ruckig", ...)   # a time law names a backend: built in or a plugin
catalog(use_case="pid_tuning", space=space)                         # what generates here, and the catalog rows not built yet
kinds()                                                             # the registry: in-tree kinds/*.py and entry-point plugins
@kind(use_cases=..., needs=..., catalog_id=..., starts_as=..., dimensions=...)   # a new kind: one decorated function, one file
LinearMotion.from_sync(pose, sync_dir, sync_speed); synchronised(space, t, target, t0=, t1=)   # the moving-target contract
track_and_release(space, target, start=meet_moving_point(space, target))   # a target object in, its data form in the recipe
summary(session)                                                    # one line per segment: span, label, call, then wait
```

## 2. What changed from PEP 0001

| | PEP 0001 (0.0.1, the splice) | PEP 0002 (0.1.0, the signal library) |
|---|---|---|
| a kind | a frozen config record class (`StepTrain(joint=...)`, V2) | a decorated plain function (`step_train(space, joint=...)`, V1) |
| a purpose plan | a robot-free `Plan` of records (`plans.pid_tuning(...)`, V3) | a **higher-order kind**: a function that chains lower kinds (`pid_tuning(space, ...)`) and is recorded as its own one call |
| the robot | a `Workspace` handed to a `JointTrajectoryGenerator` (V4) | a `JointSpace` passed as every kind's first argument, its slots injected |
| the result | a `Program` (`.trajectory`, `.segments`, `.steps`, `to_json`) | the `JointTrajectory` itself: segments are a per-sample channel plus labels in `header.meta` (V1 decision 2) |
| configuration | a program config (records + workspace) | a recipe: every call, every argument, the space's record and fingerprint |
| time laws | minimum-jerk only | minimum-jerk and trapezoid built in, any other from a plugin; legs may start or end moving |
| Cartesian | slots that gated, generating nothing | `cartesian_path`, `cartesian_waypoints`, `pick_and_place` with FK/IK passed in |
| multi-waypoint | `Move` (fractions of the room) | `move` and `joint_waypoints` (absolute configurations), `cartesian_waypoints` |
| plugins | none | entry points by role: `robot_trajectories.kinds`, `.time_laws` (`.planners`, `.optimizers` reserved) |
| collision, moving targets | catalogued, unchecked | collision slots checked by the guard; a moving-target contract and ruler |
| motion | v6 | **v6, bit for bit** (§9) |

Deleted (git history keeps them): `kind.py` (Kind, Context, param), `generator.py` (JointTrajectoryGenerator), `program.py`
(Program), `plan.py` / `plans.py` (Plan, PlanStep, step), `workspace.py` (Workspace), `kinds/repeat.py` (Repeat).

V1's own decision 1 ("a kind starts and ends at its `start`", with a leg back to start) is **not** taken: it would change v6's
motion (the pendulum session's `random_waypoints` then `home` would become one segment). A kind starts at `start` (default
home) and ends wherever its motion ends, at rest; a higher-order kind threads `start=` from one call to the next (passing the
previous JointTrajectory: its last sample), and `chain` puts in a labelled transition where a kind starts elsewhere.

## 3. Higher-order kinds

A higher-order kind is a kind whose body chains other kinds and returns the chain:

```python
@kind(use_cases=("pid_tuning", "bandwidth"), family="controller_test", starts_as="time_signal", dimensions=("1D",), higher_order=True)
def pid_tuning(space, joint=None, *, steps_fraction=STEPS_FRACTION, dwell_s=DWELL_S, chirp_hz=(0.2, 5.0), chirp_s=20.0,
               settle_s=2.0, end_settle_s=1.0, start=None):
    train = annotate(step_train(space, joint, steps_fraction=steps_fraction, dwell_s=dwell_s, start=start), "step response: ...")
    settle = annotate(hold(space, settle_s, start=train), "settle ...")
    sweep = annotate(chirp(space, joint, start_hz=chirp_hz[0], end_hz=chirp_hz[1], duration_s=chirp_s, start=settle), "...")
    rest = annotate(hold(space, end_settle_s, start=sweep), "settle before the next test", wait_after=True)
    return chain(train, settle, sweep, rest)
```

Its recipe is its own one call (`{"kind": "pid_tuning", ...}`); its segments keep the calls that make each alone, so
`split(pid_tuning(...))` gives four tapes, each rebuildable by itself. Ported from v6's plans: `pid_tuning`, `friction_id`,
`tracking_check`, `workspace_survey`, `dynamics_excitation`, `sim_to_real_tracking` (the week; the `this_week` CLI runs it),
plus `repeat_floor_tape` and `pick_and_place`. They are tagged and listed like any kind (`catalog()` marks them
higher-order).

## 4. The kind contract (frozen)

**Signature.** `f(space, <positional-or-keyword args>, *, <keyword args>, label="<words>", start=None) -> JointTrajectory`.
`space` first, `start` last with default `None`; arguments SI with the unit in the name; draws take `seed=` (an int, refused
otherwise). One file per kind in `src/robot_trajectories/kinds/`, found by importing the folder; or a module another package
names in the `robot_trajectories.kinds` entry-point group.

**Decorator.** `@kind(use_cases, needs=LIMITS, better_with=None, unchecked=(), catalog_id=None, family=None, starts_as=None,
dimensions=None, starts="at_rest", ends="at_rest", command="position", higher_order=False, until=None)`.
- `use_cases`, `needs` (hard), `better_with` (soft: need -> what the kind does without it), `unchecked` (catalog needs the
  kind does not check when missing, e.g. `collision_model`): the catalog's vocabulary (`trajectory_catalog/kind.py`), refused
  otherwise.
- `catalog_id`: the row it implements (one kind per row); `family`, `starts_as`, `dimensions` default to the row's and are
  required without one.
- `starts="moving"`: the body is handed `start` as a `State(q, qd, qdd)`; `ends="moving"`: the output ends in motion;
  `ends="anywhere"`: an event-ended kind, `until` saying the event in words; `command="effort"`: torque-commanded.

**What the decorator does, in order.**
1. Refuses a space that lacks a hard need: `MissingRung` names the input, its rung (joint limits, kinematics, dynamics,
   torque limits, or beside the ladder) and the call that adds it. A missing soft need runs the fallback and notes it.
2. Resolves `start`: `None` (home), a configuration, a JointTrajectory (its last sample), refused outside the limits.
3. Calls the body. The body returns kernel pieces (`kernels.Arrays`, `Piece` with channels or effort, a list, or
   `Motion(pieces, notes, entry)`), or a JointTrajectory it chained (a higher-order kind).
4. Joins pieces on one clock (`kernels.join`); a first piece that does not start at `start` gets a min-jerk transition,
   labelled `transition into <label>` (v6's rule).
5. **Guards** the output: inside the position, velocity and acceleration limits (and `max_effort` for a commanded effort),
   no jump between samples, at rest at both ends unless declared moving or event-ended, and, when the space carries a
   collision checker, no colliding sample (the first one is named). Tolerances are the round's ruler's (check_scenario.py G3).
   A higher-order kind is re-guarded whole (collisions only on transitions: every other segment was checked by its kind).
6. Records the call: `{"kind", every argument with defaults written out, "start"}` in `header.meta["recipe"]` beside the
   space's record and fingerprint; a moving start is recorded as its whole state; a moving target by its data form (§7); a
   callable is refused, naming `LinearMotion`.

**Segments.** Per-sample `channels["segment"]` and `channels["piece"]` (a segment owns its samples after its first; its first
is the previous one's last); `header.meta["segments"]`: per segment `label`, `call` (the call that makes it alone; null for a
transition `chain` put in), and when set `purpose`, `wait_after`, `notes`, `made_by`, `starts`, `ends`, `command`, `until`,
`assumes`, `first_sample`; `header.meta["pieces"]`: per piece `label`, `segment`. Optional `channels["tool_target"]` (N, 3)
where a Cartesian kind commanded a tool point (NaN elsewhere) and `effort` where a kind commands torque (NaN elsewhere).
`first_sample`: where a segment's own first sample differs bit for bit from the shared boundary sample (a signed zero, the
acceleration a fade ends on), the entry keeps it, so `split` gives every segment back exactly as its kind made it.

**Operations** (all pure): `chain(*trajectories, space=None)` joins (a transition segment where needed; refuses a moving
boundary whose states differ, or mixed spaces); `split(t)` cuts into rest-to-rest pieces (`chain(*split(t)) == t`, each
piece's offset in `header.stamp`, each rebuildable alone); `repeat(t, times)` is one segment of plays (the floor is one test);
`annotate(t, purpose, wait_after=)`; `rebuild(text_or_trajectory, space=None)` replays the recipe and refuses a different
space (`strict=False` regenerates anyway); `catalog(use_case=, space=)` lists implemented kinds and then every catalog row
nothing implements yet, graded available / fallback / locked on the space.

## 5. The JointSpace and its injected slots

Constructors: `from_limits(limits, home=, max_acceleration=, rate_hz=)`, `from_description(d, max_acceleration=...)` (refused
without one, naming the fix: a URDF carries no acceleration), `within(box)`. Rungs: `with_kinematics`, `with_dynamics`,
`with_reach_map` (an optional cache of the kinematics rung), beside the ladder `with_collision` (scene) and
`with_self_collision`. Readers: `room(q)` (the hole-free box), `start_at`, `state_at`, `forward`, `inverse`, `tool_points`,
`torque`, `torques`, `predict` (effort -> motion by `dynamics.forward`), `in_collision`, `status`, `record`, `describe`.

| slot | Protocol (duck-typed) | reference a real space plugs in |
|---|---|---|
| kinematics | `joint_names`; `forward(q) -> 4x4` (or xyz, or an FKResult); `inverse(pose, seed) -> q or None` (or an IKResult); optional `cache_key`, `name`, `orientation_constraint` ("none" = position only) | bam_reach `kinematics.Kinematics` (kinematics.py:107); results coerced from its FKResult / IKResult, never imported |
| dynamics | `torque(q, qd, qdd)` (or pin_utils' `inverse_dynamics`); optional `forward(q, qd, tau) -> qdd` | pin_utils `PinDynamics` |
| collision, self_collision | `in_collision(q) -> bool`; optional `distance(q) -> float` | bam_ws's MPlib-derived check: `bam_collision.acm.generate_allowed_collision_matrix` (acm.py:426; MoveIt Setup Assistant's self-collision matrix by sampling, copied from MPlib) plus Pinocchio's `computeCollisions`, behind `in_collision(q)`. It checks a finished trajectory; it never searches for one (that is a planner) |
| reach cache | `joint_names`, `joint_solutions` (P, O, J), `valid_mask` (P, O) | bam_reach `ReachMap` |

Fixtures (numpy only, `tests/fixtures/robots.py`): the pendulum's limits and a stand-in `PendulumDynamics`; the five-bar's
closed-form FK/IK of its tool point (the bench geometry); the UR5e's DH forward kinematics and damped-least-squares inverse;
`UR5eScene`, a floor plus a box checked at the DH frame origins.

## 6. Time laws, and plugins by role

`TimeLaw(shape, peak_speed_fraction= | move_s=, min_move_s=, ramp_fraction=, options=)`. A path kind takes one as its timing
argument (`Path x time law -> JointTrajectory`, the refined spec). Built in: **minimum-jerk** (v6's, unchanged) and
**trapezoid** (Pilz PTP's profile, a fixed ramp fraction). `law.leg(q0, q1, limits, rate, start_state=, end_state=,
dynamics=)` times one leg (the quintic through both states when either end moves; a trapezoid refuses a moving end by name);
`law.along(path, limits, rate, dynamics=)` follows a dense joint path (q(s), its derivatives set the duration). `dynamics` is
the space's model, for a backend that needs it; the built-in laws never read it.

Plugins are cut by **role**, not by simple vs complex (prior-art note 2026-11-18 §7): a trapezoid and TOPP-RA are the same
role; Ruckig Pro's waypoint mode is a different one.

| entry-point group | role | in, out | built in | plugins |
|---|---|---|---|---|
| `robot_trajectories.kinds` | kinds | `f(space, ..., start)` -> JointTrajectory | `kinds/*.py` | a robot's own kinds; `tests/fixtures/plugin_site` (the test plugin) |
| `robot_trajectories.time_laws` | time laws | path + limits (+ boundary states, + the space's dynamics when needed) -> timed JointTrajectory | minimum-jerk, trapezoid | Ruckig, Ruckig Pro, TOPP-RA (lane E; its torque-limited mode reads the dynamics) |
| `robot_trajectories.planners` | planners (reserved) | start, goal, the space's collision check -> joint path points (a time law then times them) | none | caelan's motion-planners, OMPL, VAMP |
| `robot_trajectories.optimizers` | optimizers (reserved) | path and timing together; register as kinds that need a model | none | Drake, cuRobo, TrajOpt, Ruckig Pro waypoints |

**The time-law protocol** (`TimeLawBackend`): `name`; `time_parameterize(path_points, limits, rate_hz, law, *,
start_state=None, end_state=None) -> JointTrajectory` sampled at the rate from t = 0, starting and ending at the path's ends in
the given states (rest by default), passing every point in order, inside `limits` (already scaled by the law's speed
fraction), optionally with a `path_progress` channel; optional `can_service(...) -> str` (why not; MoveIt's
canServiceRequest) and `needs`: a tuple of the catalog's needs, or a method `needs(law)` when they depend on the law. The
caller checks the contract (rate, ends, end velocities) and guards the result.

**The space's dynamics reach a backend that needs them** (landed 2026-11-18; it was §11's open core change). When a
backend's needs name `dynamics_model`, `_by_backend` calls `time_parameterize(..., dynamics=model)` with the model the caller
handed `leg` / `along` (as injected: `torque(q, qd, qdd)`, or pin_utils' `inverse_dynamics`); no model is a `MissingRung`
before the backend runs, and so is `torque_limits` named with no finite `max_effort`. The torque bound is the `max_effort`
already in `limits`, and the caller holds the model's torque along the result to it at every sample (an independent ruler,
as it checks the ends). A backend that does not name `dynamics_model` is never handed a model. The callers that time paths
hand `space.dynamics` on: `legs_through` (so `joint_waypoints` and `move`), `cartesian.follow` (so `cartesian_path` and every
Cartesian kind built on it) and `time_optimal_path`. TOPP-RA's torque-limited mode is the first user:
`time_optimal_path(space, path, effort_fraction=0.9)` (the `toppra` law's `effort_fraction` option) keeps the model's torque
within that fraction of `max_effort` at every sample, never faster than the kinematic timing; how it handles a model outside
TOPP-RA's rigid-body form (friction) is in the plugin's docstring.

**Rules the seam keeps.**
- **Two providers of one name are refused**, never ranked (bam_traj_gen `ruckig_backend.require_local_ruckig_pro`: Community
  and Pro both import as `ruckig`, so install order would pick the behaviour). A Pro plugin is a drop-in for the name
  `ruckig`; installing it beside Community is ambiguous and refused.
- **Ruckig Community never gets intermediate waypoints**: Community sends them to Ruckig's cloud servers. A Ruckig backend's
  `can_service` refuses more than two path points unless the local Pro build is verified (the `ruckig-pro` distribution at the
  pinned version, from the workspace checkout, the Pro-only API present, no cloud markers in the native module).
- A plugin that fails to load is reported (`plugins.errors()`, a warning); the rest load. A recipe naming a missing plugin is
  refused naming the group to install it under.

How a plugin repository registers itself (its `pyproject.toml`):

```toml
[project.entry-points."robot_trajectories.kinds"]
my_kinds = "my_package.kinds"                       # a module whose @kind functions register on import

[project.entry-points."robot_trajectories.time_laws"]
ruckig = "robot_trajectories_ruckig:RuckigBackend"  # an instance, or a class made with no arguments
```

`tests/test_plugins.py` holds this with `tests/fixtures/plugin_site/` (a kind and a time law, found through the real
entry-point mechanism in a fresh interpreter, listed and guarded like built-ins).

## 7. Moving targets (the contract; the kinds are lane F)

Zach: "Other things we can put into this library are like sync... You can account for like trajectory points that are moving".
A `MovingTarget` is `target(t) -> pose`, `velocity(t) -> twist`, `stamp`, `latency_s`, `window = (enter, leave)` and
`assumes` (the motion model in words). `LinearMotion` is the constant-velocity case and absorbs bam_traj_gen's belt chase
(`ArmParams.sync_dir` / `sync_speed`, waypoint_action.py:223; `execution_latency`). `TargetWaypoint(pose_in_target, sync)`
says which waypoints chase it (`chase`), keep the offset reached at the sync point (`inherit`, bam's `sync_at_target_time`)
or stay in the base frame (`fixed`). **A target is data in a recipe**: a kind's `target=` may be a `LinearMotion` or a robot_msgs
`PoseTrajectory` object (or LinearMotion's fields as a mapping, the first form), and the recipe writes its data form, tagged
`moving_target` (LinearMotion's fields; the PoseTrajectory in robot_msgs' own `to_dict`), which decodes to the identical
object, so `rebuild` makes the identical samples. A bare callable has no data form and is refused, naming `LinearMotion`.
**`meet_moving_point`** ends moving with the target (catalog row of the same name, `segments_join="ends_moving"`), so
`track_and_release` follows it and `chain` checks the hand-over state. Refusals are named: `CannotCatch(cause)`, code `cannot_catch` (bam_traj_gen
traj_generator.py:41), causes `target_outruns_arm`, `residual_grows`, `intercept_out_of_reach`, `time_law_error`,
`target_leaves_window`.

**Synchronised** means: within `tolerance_m` of the target at every sample from t0 to t1 (forward kinematics; orientation too
for a pose), and the tool's velocity equal to the target's at t0 within `velocity_tolerance_m_s`. `synchronised(...)` is the
ruler, and `tests/test_contract.py` shows it pass and fail (a 2 mm shift).

**A precomputed synchronised trajectory is a prediction.** Vendors track the target live in the controller (UR adds the
conveyor offset each cycle; ABB keeps the tool's speed relative to the moving work object). So a lane-F kind writes the
target's `assumes` into its segment entry, and execution must replan on new measurements (a JointTrajectoryController
replaces the running trajectory at a bridge point) or apply a controller-side offset; a monitor can check the assumption
against the encoder.

## 8. Cartesian and multi-waypoint, with FK/IK passed in

- `cartesian_path(space, points_or_poses, time_law=, orientation="hold" | "follow")`: straight tool lines through the points,
  inverse kinematics every `eef_step_m` (1 mm), refused at the first failing sample with its index, the tool point and why (no
  solution, outside the limits, a miss above tolerance, a joint jump above `max_joint_step_rad` or `jump_threshold` x the mean
  step: MoveIt's computeCartesianPath and its jump threshold, which MoveIt itself has since dropped, so the check is ours). One
  rest-to-rest leg per line, a stop at each point (a corner turned at speed is an unbounded acceleration; a curve that does
  not stop is a path kind of its own: `cartesian_arc`, lane C). Timing: the joint path's derivatives set the duration under
  the time law; positions are cubic Hermite in progress, so forward kinematics of the output tracks the line to micrometres.
- `cartesian_waypoints(space, points)`: inverse kinematics at the points only, joint-space legs between (the Cartesian twin
  of `joint_waypoints`).
- `pick_and_place(space, pick, place, lift_m=, up=)`: the U: approach (a joint move), down, dwell, up, over, down, dwell, up,
  eight segments, the dwells `wait_after`.
- `joint_waypoints(space, waypoints, time_law=, dwell_s=)`: absolute configurations, PTP legs.
- Tested on the UR5e and the five-bar (`tests/test_cartesian.py`): inside the limits, within 1 mm of every commanded tool
  point by forward kinematics, an independent line-distance oracle within 1 mm, an unreachable point refused, a line through a
  scene obstacle refused at its first colliding sample.

## 9. Bit for bit, and what is not

Held (`tests/test_equivalence.py`): the vendored kernels byte for byte; the scenario's pendulum session joined and its seven
segments, the five-bar and UR5e sessions; the week joined and its eight tapes (also against v6's own `.npz` files); every
ported kind with its defaults and every ported plan on all three robots (63 digests taken from 0.0.1 before the rewrite).

Not held, deliberately: v6's configuration **text** (the program config digests, the tapes' `.config.json`). The configuration
is now a recipe of calls (`.recipe.json`); every recipe is held to rebuild its trajectory exactly. Two choices kept the motion:
`split` restores each segment's own first sample (`first_sample`), and `repeat` copies the tape's samples where v6's `Repeat`
regenerated each play from where the last ended; the two differ only when a tape ends within 1e-9 rad of its start but not bit
for bit, which no ported tape does (the floor tape's digests hold on all three robots).

## 10. Decided here (reversible), with the undo

- **Kinds end where their motion ends, not back at `start`** (V1's decision 1 not taken; §2). Undo: add the return leg in the
  decorator and re-derive the digests.
- **`repeat` makes one segment** (the floor is read from all plays together). Undo: emit one segment per play in
  `segments.repeat`.
- **Plugin groups are named by role** (`time_laws`, reserved `planners` / `optimizers`), and two providers of one time law are
  refused rather than ranked. Undo: rename the group constants in `plugins.py` / `time_law.py`.
- **`collision_model` is a real slot**: catalog rows needing it are locked without a checker (0.0.1 listed them "not
  checked"); kinds that only benefit list it in `unchecked`. Undo: put `collision_model` back in `UNCHECKED_NEEDS`.
- **The guard checks collisions on every kind when the space has a checker**, not only on kinds that declare the need.
- **`summary(t)` and `calls(t)`** replaced `Program.summary()` / `to_python()`.
- **Seven catalog rows added** so lanes never edit the catalog: `joint_waypoints`, `cartesian_waypoints`, `cartesian_arc`,
  `approach_retreat`, `moving_target_track`, `track_and_release`, `moving_pick_and_place`; the use-case tag
  `conveyor_pick_and_place` added and set on `pick_and_place_cycle`, `touch_down_contact`, `rapid_replan` and the moving rows.
- **An eighth, `meet_moving_point`, and the join `ends_moving`** (2026-11-18, the lanes' flagged core changes): the lane F test
  stand-in promoted to a kind with its motion unchanged, so `track_and_release` has a row to follow. Undo: delete
  `kinds/meet_moving_point.py` and the row, put the stand-in back in `tests/fixtures/lane_f.py`.
- **A moving target is recorded by its data form** (`moving.target_record`), the stored direction restored exactly rather
  than normalised again. Undo: drop `has_data_form` from `registry._encode` (kinds then take the mapping form only).
- **Time-law backends see the dynamics** by a `dynamics=` keyword on `leg` / `along`, passed only to a backend whose needs
  name `dynamics_model`, and the result's torque is held to `max_effort` by the caller. Undo: remove the keyword from
  `time_law.py` and its callers, and `effort_fraction` from the TOPP-RA plugin and `time_optimal_path`.

## 11. Not built (and where it goes)

The 50 catalog rows `catalog().todo` lists, split into lanes A-F in [docs/KINDS_TODO.md](../KINDS_TODO.md); the Ruckig,
Ruckig Pro and TOPP-RA backends (lane E; TOPP-RA's torque mode landed with the dynamics seam, §6); jerk is not guarded (no built-in kind is jerk-limited); planners and
optimizers are reserved roles; smooth Cartesian curves (arcs, rasters) are lane C/D kinds on `cartesian.follow`.

## 12. Migration (who moves onto this, later)

sim2real `schedule/joint.py`'s `Family` + `register(...)` is already this pattern: a family becomes `@kind`, `make(description,
rng, stretch)` becomes `f(space, ..., seed=, start=)`. bam_rack_twin `schedule.py`'s families map onto `linear_chirp`, `chirp`,
`velocity_plateaus`, `step_train`, `repeat_floor_tape` and `deployment_motion_classes`. traj_integration_tests' `Segment`
becomes a kind and `_sample_segments` becomes `chain`. bam_traj_gen's moving-target chase moves onto §7's contract (lane F).
None of that is done here.
