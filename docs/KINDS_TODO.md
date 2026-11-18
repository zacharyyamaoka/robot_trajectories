# Kinds to add: the contract the lanes work to

Every catalog row nothing implements yet (50 of 68), split into six lanes that can run in parallel **without touching a core
file**. `catalog().todo` lists the same 50 rows live; this file says who builds each, where, on which robots, and how it is
tested. The contract every kind obeys is [PEP 0002 §4](peps/0002-signal-library.md#4-the-kind-contract-frozen); read it first.

| lane | what | rows | depends on |
|---|---|---|---|
| **A** | excitation + controller tests (position-commanded) | 10 | nothing |
| **B** | quasi-static + hold + stress, and every torque / event-ended kind | 15 | nothing |
| **C** | point-to-point + coverage (incl. CIRC and approach/retreat of the 6-DOF family) | 11 | nothing |
| **D** | standard tests + calibration + task cycles | 9 | C's `cartesian_arc` for `circular_test` / `iso9283_path` (or its own arc helper) |
| **E** | time-law plugin packages under `plugins/` (Ruckig, Ruckig Pro, TOPP-RA) and the two rows they serve | 2 | nothing |
| **F** | moving-target (synchronised) kinds | 3 | nothing (the contract is `robot_trajectories.moving`) |

## Rules of the road (every lane)

**A lane may create, and only create:**
- `src/robot_trajectories/kinds/<function>.py`: one file per kind, found automatically;
- one private helper `src/robot_trajectories/kinds/_lane_<x>.py`, shared by that lane's kinds (imported, registers nothing);
- `tests/test_lane_<x>.py`: that lane's tests (the template is below);
- `tests/fixtures/lane_<x>.py`: that lane's own fixtures (numpy only; stand-ins labelled as such);
- lane E only: `plugins/<package>/...` (its own `pyproject.toml`, source, tests).

**A lane must not edit** `registry.py`, `segments.py`, `space.py`, `time_law.py`, `cartesian.py`, `moving.py`, `plugins.py`,
`recipe.py`, `kernels.py` (byte-identical to the round's), `trajectory_catalog/` (every row below already exists),
`tests/fixtures/robots.py`, `tests/kind_contract.py`, or any existing test. **If a kind needs a core change, stop and report
it**: a change to a contract other lanes import is a barrier, landed first and alone by the dispatcher.

**Every kind:**
- is `@kind(use_cases=..., needs=..., catalog_id="<row id>")` on `f(space, ..., start=None)`;
- takes SI arguments with the unit in the name (`_s`, `_hz`, `_rad`, `_m`, `_fraction`), a `label="<words>"` keyword, and a
  `seed=` int wherever it draws;
- **is task-agnostic** (Zach, "Robot Trajectories", 2026-11-18): no task words in its name, parameters or labels (no grasp,
  gripper, belt, conveyor, item, object, bin). What it serves is a use-case tag;
- asks the space for its room (`space.room(start)`), never its own box, so a reach map keeps it reachable;
- keeps a `# WHY:` comment wherever it preserves a stated choice (a source's number, a refusal, a cap).

**Three output shapes that are not plain position trajectories, and how each still returns a JointTrajectory:**

1. **Pose sets** (`output="pose_set"`: stations to visit, dwell and measure at). Return the stations joined by
   `TimeLaw` legs (`time_law=` argument, default minimum-jerk) with a dwell piece at each, labelled `"<label>: station k"`;
   put the stations' configurations in the segment entry through `Motion(pieces, notes, entry={"stations": [[...], ...]})`.
   With kinematics, a station given as a tool pose is solved with `cartesian.solve_points`.
2. **Torque profiles** (`output="torque_profile"`, `space="torque"`). Declare `command="effort"` and the needs
   `torque_limits` + `dynamics_model`. Return `Piece(t, q, qd, qdd, label, effort=effort)`: `effort` (N, J) is the
   command; `q, qd, qdd` are the prediction `space.predict(start, effort)` (the model's `forward(q, qd, tau)`). The guard
   checks `effort` against `max_effort` and the prediction against the position limits. A profile that leaves the joint
   moving declares `ends="anywhere"`. Without a `forward` model the kind is locked (`MissingRung`), never guessed.
3. **Event-ended command sequences** (`output="command_sequence"`: a seek, a release, a relay). The JointTrajectory is the
   nominal plan up to the expected event plus a margin, and the kind declares `ends="anywhere"` and
   `until="<the event in words, with its numbers>"` (e.g. `"stall: |qd| < 0.01 rad/s for 0.2 s, or 0.05 rad past the
   expected stop"`), so a player stops on the event. Torque-commanded ones are also `command="effort"`.

**Moving boundaries** (a kind that hands a moving state on, lane F mainly): declare `ends="moving"` / `starts="moving"`; a
moving-start body receives `start` as a `State(q, qd, qdd)`. `TimeLaw.minimum_jerk(...).leg(q0, q1, ..., start_state=,
end_state=)` is the quintic through both states. `chain` refuses a moving end joined to anything that does not start in
that exact state.

## The 6-DOF arm family on the UR5e (Pilz industrial motion planner's set and its neighbours)

| motion | kind | status |
|---|---|---|
| PTP (joint move, synchronised trapezoid) | `move` / `joint_waypoints` with `TimeLaw.trapezoid(...)`; lane C's `trapezoidal_velocity` | built; the named row is lane C |
| LIN (straight tool line) | `cartesian_path` | built |
| CIRC (tool arc by centre or interim point) | `cartesian_arc` | lane C |
| approach / retreat along the tool axis | `approach_retreat` | lane C |
| joint waypoints | `joint_waypoints` | built |
| Cartesian waypoints | `cartesian_waypoints` | built |
| blended sequence (through, not stopping at, via points) | `via_point_spline` | lane C |
| time-optimal along a path | `time_optimal_path` (TOPP-RA plugin) | lane E |
| jerk-limited S-curve | `jerk_limited_scurve` (Ruckig plugin) | lane E |
| pick-and-place U | `pick_and_place` (static points); `moving_pick_and_place` (a moving first point) | built; moving is lane F |

## Robots, in the tests

`tests/scenario.py` and `tests/fixtures/robots.py` give: the **pendulum** (1D, `scenario.pendulum_space()`, CAN 11's
limits; `robots.PendulumDynamics()` for torque kinds, with a stand-in `max_effort` such as 1.0 N·m added in the test), the
**five-bar** (2D, `scenario.fivebar_space()` with its reach map; `robots.FiveBarKinematics()` for Cartesian kinds, planar,
position only; start Cartesian tests at `np.radians([128.0, -126.0])`, inside the reach map away from the elbow
singularity; plane "up" is `(0, -1, 0)`), and the **UR5e** (3D/6D, `scenario.ur5e_space()`;
`robots.UR5eKinematics()` for Cartesian kinds; `robots.UR5eScene(box_min, box_max)` as a collision scene).
Every Cartesian test also holds the output to its commanded tool points by forward kinematics (within 1 mm), as
`tests/test_cartesian.py` does, and refuses an unreachable point.

## Lane A: excitation + controller tests (10)

| row (priority) | function → file | rung (beyond joint limits) | robots | notes |
|---|---|---|---|---|
| `slow_plus_fast_sines` (next) | `slow_plus_fast_sines` → `kinds/slow_plus_fast_sines.py` | none | 1D pendulum; one joint of five-bar, UR5e | amplitudes solved like `kernels._scaled` |
| `stepped_sine_dwell` (next) | `stepped_sine_dwell` → `kinds/stepped_sine_dwell.py` | none | 1D; one joint of others | a chain of `sine_dwell`s with settle holds: a higher-order kind |
| `orthogonal_multisine` (later) | `orthogonal_multisine` → `kinds/orthogonal_multisine.py` | collision checked when the space has a checker (`unchecked=("collision_model",)`) | 2D five-bar, 3D/6D UR5e | each joint its own line set (Morelli) |
| `prbs` (later) | `prbs` → `kinds/prbs.py` | none | 1D; one joint of others | position steps band-limited by min-jerk transitions; `seed=` |
| `band_limited_noise` (later) | `band_limited_noise` → `kinds/band_limited_noise.py` | none | 1D; one joint of others | filtered in the frequency domain, faded; `seed=` |
| `fourier_series_excitation` (next) | `fourier_series_excitation` → `kinds/fourier_series_excitation.py` | dynamics (`dynamics_model`): the optimisation's objective | 2D five-bar, 6D UR5e (with a lane-A fixture dynamics) | without dynamics: locked, never an unoptimised series |
| `biased_dither` (next) | `biased_dither` → `kinds/biased_dither.py` | none | 1D; one joint of others | slides without reversing |
| `staircase` (next) | `staircase` → `kinds/staircase.py` | none | 1D; one joint of others | levels up then down, dwell at each (ISO 230-2 bidirectional) |
| `doublet_3211` (later) | `doublet_3211` → `kinds/doublet_3211.py` | none | 1D; one joint of others | pulse widths in time; position-limited |
| `rapid_replan` (later) | `rapid_replan` → `kinds/rapid_replan.py` | named home (`home_pose`) | 2D five-bar, 6D UR5e | each replan starts from the last one's state mid-motion: `ends="moving"` pieces joined by quintic legs |

## Lane B: quasi-static + hold + stress, and every torque / event-ended kind (15)

| row (priority) | function → file | rung | robots | output contract |
|---|---|---|---|---|
| `torque_chirp` (next) | `torque_chirp` → `kinds/torque_chirp.py` | torque limits + dynamics (`forward`) | 1D pendulum + `PendulumDynamics` | effort profile; `command="effort"`; `ends="anywhere"` unless it fades to rest |
| `torque_steps` (later) | `torque_steps` → `kinds/torque_steps.py` | torque limits + dynamics | 1D pendulum | effort profile |
| `free_swing_release` (next) | `free_swing_release` → `kinds/free_swing_release.py` | joint limits (+ dynamics to predict the swing) | 1D pendulum | lift by position, then release: `until="released: kp = kd = 0 after ..."`, `ends="anywhere"` |
| `relay_feedback` (later) | `relay_feedback` → `kinds/relay_feedback.py` | torque limits | 1D pendulum | relay on the measured error: nominal plan + `until`, `command="effort"` |
| `breakaway_torque_ramp` (next) | `breakaway_torque_ramp` → `kinds/breakaway_torque_ramp.py` | torque limits | 1D pendulum | ramp until motion: `until="breakaway: |qd| > ..."` |
| `small_reversals` (next) | `small_reversals` → `kinds/small_reversals.py` | joint limits (the camera measures; not an input) | 1D; one joint of others | position-commanded |
| `locked_output_torque_ramp` (later) | `locked_output_torque_ramp` → `kinds/locked_output_torque_ramp.py` | torque limits | 1D pendulum (output locked: positions held) | effort profile; positions = start held, stated in `notes` |
| `quasi_static_pose_holds` (next) | `quasi_static_pose_holds` → `kinds/quasi_static_pose_holds.py` | kinematics + dynamics | 2D five-bar, 6D UR5e (lane-B fixture dynamics) | pose set (stations + dwells, dithered) |
| `hard_stop_seek` (next) | `hard_stop_seek` → `kinds/hard_stop_seek.py` | torque limits | 1D pendulum (per joint in turn on others) | slow seek past the expected stop: `until="stall ..."`, `ends="anywhere"`; refuses without `max_effort` |
| `burn_in_wiggle` (next) | `burn_in_wiggle` → `kinds/burn_in_wiggle.py` | named home | 1D; per joint on others | position-commanded; collision checked when present |
| `heating_run` (later) | `heating_run` → `kinds/heating_run.py` | torque limits | 1D pendulum | effort profile, long; `until="winding at <temperature> or ..."` |
| `endurance_duty_cycle` (later) | `endurance_duty_cycle` → `kinds/endurance_duty_cycle.py` | named home | 2D five-bar, 6D UR5e | a higher-order kind: `repeat` of a duty cycle |
| `power_up_random_rests` (later) | `power_up_random_rests` → `kinds/power_up_random_rests.py` | kinematics | 2D five-bar, 6D UR5e | pose set of seeded rests (each a station) |
| `disturbance_push` (later) | `disturbance_push` → `kinds/disturbance_push.py` | named home | 2D five-bar, 6D UR5e | a plain hold; the push schedule goes in `entry={"push": ...}` |
| `weightless_float` (next) | `weightless_float` → `kinds/weightless_float.py` | torque limits + kinematics + dynamics | 2D five-bar, 6D UR5e (lane-B fixture dynamics) | hold, then `command="effort"` gravity torque only: `ends="anywhere"` |

## Lane C: point-to-point + coverage (11)

| row (priority) | function → file | rung | robots | notes |
|---|---|---|---|---|
| `trapezoidal_velocity` (later) | `trapezoidal_velocity` → `kinds/trapezoidal_velocity.py` | none | 1D pendulum, 2D, 6D | a move under `TimeLaw.trapezoid` (Pilz PTP): built in, a few lines |
| `polynomial_move` (later) | `polynomial_move` → `kinds/polynomial_move.py` | none | 1D, 2D, 6D | cubic or quintic between states; moving ends via `TimeLaw.minimum_jerk(...).leg(..., start_state=, end_state=)` |
| `via_point_spline` (next) | `via_point_spline` → `kinds/via_point_spline.py` | jerk limits (when stated) | 2D five-bar, 6D UR5e | through via points without stopping; the blended Pilz sequence |
| `cartesian_arc` (next) | `cartesian_arc` → `kinds/cartesian_arc.py` | kinematics | five-bar (planar arc), UR5e | Pilz CIRC: a path `s -> pose` given to `cartesian.follow` (the arc is the geometry; IK every step) |
| `approach_retreat` (next) | `approach_retreat` → `kinds/approach_retreat.py` | kinematics | five-bar, UR5e | a `cartesian.line` along the tool's own z (or a given axis), out and back |
| `cartesian_a_to_b` (next) | `cartesian_a_to_b` → `kinds/cartesian_a_to_b.py` | kinematics + reach map (as a cache) | five-bar, UR5e | poses sampled from the reach set, IK at the ends, a joint-space leg |
| `input_shaped_move` (later) | `input_shaped_move` → `kinds/input_shaped_move.py` | dynamics | 1D, 2D, 6D | ZV / ZVD convolved move; the mode from arguments, never guessed |
| `cartesian_raster` (next) | `cartesian_raster` → `kinds/cartesian_raster.py` | kinematics (+ reach map as a cache) | five-bar (the plane), UR5e | lines through `cartesian.follow`, stepping over between rows |
| `space_filling_pose_set` (next) | `space_filling_pose_set` → `kinds/space_filling_pose_set.py` | none (collision when present) | 2D, 6D | pose set: Sobol / Halton / Latin hypercube in the room, numpy only, `seed=` |
| `lissajous_multi_joint` (later) | `lissajous_multi_joint` → `kinds/lissajous_multi_joint.py` | none | 2D, 6D | incommensurate sines per joint, faded |
| `coverage_greedy_swings` (next) | `coverage_greedy_swings` → `kinds/coverage_greedy_swings.py` | none | 1D; one joint of others | swings picked to fill (q, qd) cells |

## Lane D: standard tests + calibration + task cycles (9)

| row (priority) | function → file | rung | robots | notes |
|---|---|---|---|---|
| `iso9283_pose_cube` (next) | `iso9283_pose_cube` → `kinds/iso9283_pose_cube.py` | kinematics | UR5e (3D/6D) | pose set P1-P5 in the largest cube inside the reach; repeats per the standard |
| `iso9283_path` (later) | `iso9283_path` → `kinds/iso9283_path.py` | kinematics | five-bar (line, circle), UR5e | lines through `cartesian.follow`; the circle via lane C's arc or a private helper |
| `circular_test` (later) | `circular_test` → `kinds/circular_test.py` | kinematics | five-bar, UR5e | a closed circle (ballbar), both directions |
| `kinematic_calibration_pose_set` (next) | `kinematic_calibration_pose_set` → `kinds/kinematic_calibration_pose_set.py` | kinematics | 2D five-bar, 6D UR5e | pose set chosen by an observability index (numpy) |
| `hand_eye_station_set` (next) | `hand_eye_station_set` → `kinds/hand_eye_station_set.py` | kinematics | UR5e (6D) | pose set of tool poses around a point, solved with `solve_points` |
| `camera_board_views` (later) | `camera_board_views` → `kinds/camera_board_views.py` | kinematics | UR5e (6D) | pose set |
| `return_to_ready` (next) | `return_to_ready` → `kinds/return_to_ready.py` | kinematics + named home | 2D five-bar, 6D UR5e | joint waypoints that keep margins to READY; collision checked when present |
| `adept_cycle` (later) | `adept_cycle` → `kinds/adept_cycle.py` | kinematics | five-bar (the plane), UR5e | up / across / down as `cartesian_path` lines, as fast as the limits allow |
| `touch_down_contact` (later) | `touch_down_contact` → `kinds/touch_down_contact.py` | kinematics | five-bar, UR5e | descend, the last stretch at a set tool speed aiming below the surface, dwell, lift |

## Lane E: time-law plugin packages (2 rows + 3 packages)

Each package lives in `plugins/<package>/` with its own `pyproject.toml`, registers through the entry-point group
`robot_trajectories.time_laws`, depends on `robot-trajectories` and the library it wraps, and has tests that hold its
backend to the `TimeLawBackend` protocol (time_law.py) on the pendulum, the five-bar and the UR5e: starts and ends at the
path's ends in the given states, sampled at the rate from t = 0, inside the limits it was handed, `path_progress` when it
follows a dense path, and `can_service` refusals by name. Install for the tests with
`uv pip install -e plugins/<package>` into a separate environment; the core venv stays numpy + robot_msgs.

```toml
[project.entry-points."robot_trajectories.time_laws"]
ruckig = "robot_trajectories_ruckig:RuckigBackend"
```

| package | registers | what | the rule it must keep |
|---|---|---|---|
| `plugins/robot_trajectories_ruckig/` | `ruckig` | Ruckig **Community** (PyPI `ruckig`): state-to-state legs, boundary states (target velocity / acceleration), jerk from `max_jerk` | **never intermediate waypoints**: Community sends them to Ruckig's cloud. `can_service` refuses more than two path points and names the Pro package |
| `plugins/robot_trajectories_ruckig_pro/` | `ruckig` (a drop-in, same name) | Ruckig **Pro**, the local build: intermediate waypoints (`WaypointsCalculator`), position limits | on load, verify provenance as bam_ws `bam_traj_gen/ruckig_backend.require_local_ruckig_pro` does (the `ruckig-pro` distribution at the pinned version, from the workspace checkout's `direct_url.json`, the Pro-only API present, no cloud markers in the native module) and refuse to register otherwise; never installed beside Community (`time_law.backend` refuses two providers of one name) |
| `plugins/robot_trajectories_toppra/` | `toppra` | TOPP-RA: time-optimal along a dense joint path; boundary path speeds; torque-limited mode (`effort_fraction`) | the torque mode reads the space's dynamics through the seam's `dynamics=` (landed 2026-11-18, PEP 0002 §6): `needs(law)` names `dynamics_model` and `torque_limits`, the model's torque stays within `max_effort` at every sample |

| row | function → file | rung | robots | notes |
|---|---|---|---|---|
| `jerk_limited_scurve` (next) | `jerk_limited_scurve` → `kinds/jerk_limited_scurve.py` | jerk limits + the `ruckig` plugin | 1D, 2D, 6D | build `TimeLaw("ruckig", ...)` inside the body (a default argument would be evaluated at import, before plugins load); without the plugin the call is refused naming the group |
| `time_optimal_path` (later) | `time_optimal_path` → `kinds/time_optimal_path.py` | the `toppra` plugin | 1D, 2D, 6D | `TimeLaw("toppra", ...).along(path, ...)` |

## Lane F: moving-target (synchronised) kinds (3)

The contract is `robot_trajectories.moving` (PEP 0002 §7): a `MovingTarget` (`target(t)`, `velocity(t)`, `stamp`,
`latency_s`, `window`, `assumes`; `LinearMotion` is the constant-velocity case and absorbs bam_traj_gen's
`sync_dir` / `sync_speed`), `TargetWaypoint` (`chase` / `inherit` / `fixed`), `CannotCatch(cause)` refusals, and the ruler
`synchronised(space, trajectory, target, t0=, t1=)`. A lane-F kind takes `target=` (a `LinearMotion` or `PoseTrajectory`
object, recorded by its data form, or LinearMotion's fields as a mapping, or a fixed pose) and a sync window, starts and ends at rest, writes the target's `assumes` into its segment entry
(`Motion(..., entry={"assumes": target.assumes, "sync": [t0, t1]})`), and refuses with `CannotCatch` (never a silent
miss). Shared helpers go in `kinds/_lane_f.py`: a numeric Jacobian from `space.forward` (the joint velocity that gives the
target's twist at the sync point), and a follower that solves IK at every sample of a time-given tool path.

| row (priority) | function → file | rung | robots | notes |
|---|---|---|---|---|
| `moving_target_track` (next) | `moving_target_track` → `kinds/moving_target_track.py` | kinematics | five-bar (a point moving in the plane), UR5e | rest → quintic leg ending at `target(t0)` with matched velocity (`end_state=`) → track to t1 → quintic to rest |
| `meet_moving_point` (next, built) | `meet_moving_point` → `kinds/meet_moving_point.py` | kinematics | five-bar, UR5e | `ends="moving"`: `moving_target_track` without the coast, the state `track_and_release` continues (promoted from the lane-F test stand-in 2026-11-18) |
| `track_and_release` (later) | `track_and_release` → `kinds/track_and_release.py` | kinematics | five-bar, UR5e | `starts="moving"`: continues a tracked state, dwells in the target's frame, leaves along a direction to rest |
| `moving_pick_and_place` (next) | `moving_pick_and_place` → `kinds/moving_pick_and_place.py` | kinematics | UR5e (3D/6D) | `pick_and_place`'s U where the first point moves: down / dwell / up chase it (`TargetWaypoint(sync="chase")`), the leg after inherits the offset |

Tests: `synchronised(...)` passes over the window, and fails when the target is shifted 2 mm (the ruler must be able to
fail); `CannotCatch("target_outruns_arm")` when the target is faster than the arm; `CannotCatch("target_leaves_window")`
when t0 lies outside `target.window`.

## Test templates

Lanes A-D and F (one file per lane, `tests/test_lane_<x>.py`, runnable as a plain script while the arm is powered):

```python
"""Lane A's kinds, each held to the kind contract on the robots it applies to, and to what it is for."""

from __future__ import annotations

import numpy as np

import _runner
import scenario
from kind_contract import check_kind
from robot_trajectories import prbs


def test_prbs():
    for space, joint in ((scenario.pendulum_space(), "can11"), (scenario.fivebar_space(), "j2"), (scenario.ur5e_space(), "elbow_joint")):
        check_kind(prbs(space, joint, seed=3), space)
    # + one claim about what it is FOR, that can fail: e.g. its spectrum is flat to f_max within 3 dB


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_lane_a"))
```

A Cartesian kind adds the 1 mm check and an unreachable refusal (copy `tracks` / `off_line` from `tests/test_cartesian.py`
into the lane's test file); a torque kind builds its space with
`JointSpace.from_limits(robots.PENDULUM_LIMITS.replace(max_effort=np.array([1.0]))).with_dynamics(robots.PendulumDynamics())`
and asserts `segment_info(t)[0]["command"] == "effort"` and that an effort over `max_effort` is refused
(`tests/test_contract.py`'s `contract_torque_pulse` is the worked example); a moving-target kind asserts `synchronised`.

Run each file as a script:

```bash
cd ~/robot_trajectories && /home/bam/robot_trajectories/.venv/bin/python tests/test_lane_a.py
```
