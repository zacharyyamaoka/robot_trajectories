"""One kind of trajectory the planned JointTrajectoryGenerator can make: the record every catalog entry is.

Zach (Daily Note, 2026-11-18): "There are lots of fun different types of trajectories like the workspace sweap, the chirps, the steps
... Some trajectories are good for testing like PIDs ... I want like a library or Catalog of all the different trajectories."

WHY a frozen dataclass and not JSON rows (Zach's rule, vault memory feedback_python_ground_truth): hand-authored records are Python
frozen dataclasses, one per .py; ``catalog.json`` is generated from them by ``make_json.py`` and never edited.

WHY every vocabulary is a closed tuple here: the landing page filters and the generator's entry points (joint limits only, a ReachMap,
a full RobotDescription) group kinds by these tags, so a tag that is not in the vocabulary is a typo, refused at construction.
Units are SI throughout (rad, rad/s, s, Hz, N.m, m); angles are shown to people in degrees at the display edge, never stored so.

Paths and trajectories (added 2026-11-18). Zach, Daily Note 2026-11-18 "Further thoughts": "We are either defining trajectories in
joint or cartesian space AND then we basically need to know the limits for both of those spaces ... it's almost like a PathGenerator
as well ... all joint trajectories you can also see as paths ... two sides of the same coin ... we can make it clear what are the ones
that start out as paths", and "I want to be able to see like the catalog at a glance .... 1D, 2D, 3D, 6D paths / trajectories". Four
tags answer that, each a closed vocabulary below:

- ``starts_as`` (authored): ``time_signal`` when the time law IS the content (a chirp, a multisine, a step train, a torque profile, a
  hold: re-timing it changes what it measures) or ``path`` when the geometry is the content and the timing is a separate choice (a
  speed ladder, a grid snake, a raster, a circle, waypoints, pose sets). This is the path-velocity decomposition: plan the geometric
  path first, then time it along the path (Kant & Zucker 1986, IJRR 5(3), https://doi.org/10.1177/027836498600500304), which is how
  time-parameterisers split the job today: TOPP-RA (Pham & Pham 2018, https://doi.org/10.1109/TRO.2018.2819195) and Ruckig
  (Berscheid & Kroger 2021, https://doi.org/10.15607/RSS.2021.XVII.015) time a given path, and MoveIt runs its planner first and a
  time-parameterisation stage after it
  (https://moveit.picknik.ai/main/doc/examples/time_parameterization/time_parameterization_tutorial.html). A path kind can take any
  time law; a time-signal kind's time law is not negotiable. A kind whose only timing knob is a speed you pick (a tool speed, a feed,
  a speed rung) is still a path; it is a time signal when the SHAPE of its time law is what it measures (frequency content, step
  sharpness, plateau length, a torque or current schedule, an event schedule).
- ``defined_in`` (derived): the space its geometry or signal is written in, ``joint`` / ``cartesian`` / ``torque``. It IS the
  ``space`` field, exposed under the name the at-a-glance page uses (WHY not a second field: two fields could disagree).
- ``dimensions`` (derived for joint and torque kinds, authored for Cartesian ones), a contiguous run of ``DIMENSIONS``:
    joint / torque space, how many joints it moves at once on the round's robots: ``single_joint`` and ``per_joint_sequential``
    (one joint at a time) are ``("1D",)``; ``coordinated_multi_joint`` is ``("2D", "3D", "6D")``: every joint of the 2-DOF
    five-bar (2D) and of the 6-DOF UR5e (6D), and 3D because each such kind is written for any joint count, so it is meaningful on
    a 3-joint subset (the BAM arm's J1 + five-bar, the UR5e's base, shoulder and elbow, as the round's 3 x 3 x 3 grid does);
    Cartesian space, the geometry: a straight line ``1D``, a planar circle or raster ``2D``, a volume (a helix, a 3-D grid, the ISO
    cube) ``3D``, anything that also prescribes the tool's orientation ``6D``.
  WHY contiguous: the at-a-glance page draws a kind as one chip spanning its columns, so a gap (2D and 6D but not 3D) would be drawn
  as a claim it does not make.
- ``entry_level`` (derived from ``needs``): the lowest rung of Zach's input ladder that can generate the kind, so passing that much
  is enough ("if you just pass in joint limits that's probably enough for joint trajectories. Then if you pass in IK/FK now you can
  do cartesian paths (basically what MoveIt does)"). The ladder is cumulative, each rung including the ones before it:
    ``joint_limits``   position, velocity, acceleration, jerk limits and named rests (home_pose): a joint vector and its box;
    ``kinematics``     + forward / inverse kinematics; a ``reach_map`` maps here because FK/IK is what builds one: the generator
                       can sample reachability online and cache it, so a ReachMap becomes an optional cache, not an entry point
                       ("this can work without the reach map, doing things more online (perhaps it can cache its own results)");
    ``dynamics``       + a dynamics model (gravity, inertia, friction);
    ``torque_limits``  + the drive's torque cap (open-loop torque kinds, torque-feasibility checks).
  ``entry_extras`` (derived) holds what sits beside the ladder, not on it: ``collision`` when the kind checks a collision model
  (it rides on the kinematics, but a 1-DOF rig has nothing to collide, so it is not a rung). ``camera`` is a measurement need, not a
  generator input: it changes neither.

This module is stdlib only, so the catalog loads anywhere (a cloud VM, the landing-page builder) without the robot stack.
"""

from __future__ import annotations

from dataclasses import dataclass

# What the kind is FOR, in one word: the landing page's first grouping.
FAMILIES: tuple[str, ...] = (
    "excitation",       # broadband or swept signals that identify a plant or measure a frequency response
    "controller_test",  # inputs a controller is judged or tuned on: steps, staircases, doublets, relay, replans
    "quasi_static",     # slow motions that isolate one effect: friction plateaus, gravity triangles, reversals, breakaway
    "point_to_point",   # smooth rest-to-rest moves between given configurations or poses (the building blocks)
    "coverage",         # visit a space: joint-space grids, workspace rasters, random or low-discrepancy targets
    "standard_test",    # named, repeatable test protocols: repeat floor, speed ladder, ISO 9283 / ISO 230 tests
    "calibration",      # pose sets and sequences that calibrate kinematics, cameras, hand-eye, zeros (homing)
    "stress",           # burn-in, heating, endurance and power-up robustness
    "task_cycle",       # deployment-shaped motion: pick-and-place cycles, benchmark cycles, contact approaches
    "hold",             # hold a pose: settle and read a stop, reject a push, float weightless
)

# The FIXED use-case vocabulary (descriptions live in catalog.USE_CASES; the test holds the two equal).
USE_CASE_TAGS: tuple[str, ...] = (
    "pid_tuning",
    "bandwidth",
    "delay_id",
    "inertia_id",
    "friction_id",
    "gravity_id",
    "backlash_id",
    "full_dynamics_id",
    "tracking_eval",
    "stop_accuracy",
    "state_coverage",
    "workspace_coverage",
    "repeatability",
    "accuracy",
    "kinematic_calibration",
    "hand_eye_calibration",
    "camera_calibration",
    "homing",
    "thermal",
    "endurance",
    "speed_limits",
    "disturbance_rejection",
    "contact_eval",
    "smoke_test",
    "collision_check",
    "cycle_time",
    "demo",
)

# Where the kind is authored. A torque kind is NOT a position JointTrajectory: it is played open loop (kp = kd = 0, or a current).
SPACES: tuple[str, ...] = ("joint", "cartesian", "torque")

# How many joints move and how.
SCOPES: tuple[str, ...] = ("single_joint", "per_joint_sequential", "coordinated_multi_joint", "cartesian_path")

# What the generator hands back. WHY a fourth value beyond the brief's three spaces (2026-11-18 sweep): homing seeks, breakaway ramps and
# free swings end on an EVENT (a stall, a settle, a release), not at a time, so they are robot_msgs JointCommandSequences; calibration sets
# are a list of stations to visit and dwell at, joined by point_to_point moves.
OUTPUTS: tuple[str, ...] = (
    "joint_trajectory",  # time-indexed q (qd, qdd, effort) per joint: robot_msgs JointTrajectory
    "torque_profile",    # time-indexed torque per joint, played with no position loop
    "command_sequence",  # ordered steps with done-when conditions: robot_msgs JointCommandSequence
    "pose_set",          # stations (joint or tool poses) to visit, dwell and measure at; the moves between are point_to_point
)

# What the generator must be given. The clusters decide the entry points (see the vault note's API observations).
NEEDS: tuple[str, ...] = (
    "joint_limits",          # position range per joint
    "velocity_limits",
    "acceleration_limits",
    "jerk_limits",
    "torque_limits",         # the drive's cap, for open-loop torque kinds and torque-feasibility checks
    "reach_map",             # the reachable workspace (bam_reach ReachMap / reachable cells)
    "forward_kinematics",
    "inverse_kinematics",
    "dynamics_model",        # gravity / inertia / friction: torque prediction or experiment optimisation
    "collision_model",       # self- and scene-collision checks of every sample
    "home_pose",             # a named rest (READY / home) to start from and return to
    "camera",                # a camera or other external tracker that sees the output
)

# How a kind's start and end tie to its neighbours when the generator joins segments into one long trajectory.
JOINS: tuple[str, ...] = (
    "starts_and_ends_at_rest",    # zero velocity at both ends (anywhere): joins by a rest-to-rest move
    "periodic_returns_to_start",  # ends where it started, at rest, after whole periods (faded in and out)
    "ends_anywhere",              # ends wherever physics leaves it (a coast, a stall, a replan): the next segment starts from the measurement
)

PRIORITIES: tuple[str, ...] = (
    "now",    # needed this week for the 1-DOF pendulum tracking eval and PID tuning
    "next",   # needed for the multi-joint arm or identification after the pendulum
    "later",  # worth supporting, nobody waiting on it
)

# What the kind starts out as (see the module docstring: the path-velocity decomposition).
STARTS_AS: tuple[str, ...] = (
    "time_signal",  # the time law is the content: re-timing it changes what it measures
    "path",         # the geometry is the content: the timing is a separate choice (any time law, any speed)
)

# The space a kind's geometry or signal is written in: the SPACES vocabulary under the at-a-glance page's name.
DEFINED_IN: tuple[str, ...] = SPACES

# How many dimensions a kind lives in (joint count for joint / torque kinds, the geometry for Cartesian ones), in column order.
DIMENSIONS: tuple[str, ...] = ("1D", "2D", "3D", "6D")

# WHY derived and not authored for joint / torque kinds: the joint count is what ``scope`` already says, so a second hand-typed copy
# could only disagree with it.
DIMENSIONS_OF_SCOPE: dict[str, tuple[str, ...]] = {
    "single_joint": ("1D",),
    "per_joint_sequential": ("1D",),  # every joint in turn, one at a time
    "coordinated_multi_joint": ("2D", "3D", "6D"),
}

# Zach's input ladder, cumulative: each rung includes every rung before it (the module docstring has the reasons).
ENTRY_LEVELS: tuple[str, ...] = ("joint_limits", "kinematics", "dynamics", "torque_limits")

# What a kind may need beside the ladder.
ENTRY_EXTRAS: tuple[str, ...] = ("collision",)

# The rung each need puts a kind on; None = no rung (a measurement need, or an extra). The test holds this to cover NEEDS exactly.
ENTRY_LEVEL_OF_NEED: dict[str, str | None] = {
    "joint_limits": "joint_limits",
    "velocity_limits": "joint_limits",
    "acceleration_limits": "joint_limits",
    "jerk_limits": "joint_limits",
    "home_pose": "joint_limits",          # a named joint vector (READY), no model needed
    "forward_kinematics": "kinematics",
    "inverse_kinematics": "kinematics",
    "reach_map": "kinematics",            # built from FK/IK: an optional cache of the kinematics rung, not a rung of its own
    "dynamics_model": "dynamics",
    "torque_limits": "torque_limits",
    "collision_model": None,              # -> entry_extras "collision"
    "camera": None,                       # measures the output; the generator never needs it
}
ENTRY_EXTRA_OF_NEED: dict[str, str] = {"collision_model": "collision"}

Parameter = tuple[str, str, float | int | str]  # (name, SI unit or "1" / "count" / "text", default)
Citation = tuple[str, str]  # (short citation, URL)


def _refuse_unknown(field_name: str, values: tuple[str, ...], vocabulary: tuple[str, ...], kind_id: str) -> None:
    unknown = [value for value in values if value not in vocabulary]
    if unknown:
        raise ValueError(f"{kind_id}: {field_name} {unknown} not in the vocabulary {vocabulary}")


@dataclass(frozen=True, slots=True)
class TrajectoryKind:
    """One trajectory family worth supporting: what it is, what it serves, what it needs, where it exists today and its prior art."""

    id: str                              # snake_case, unique in the catalog
    name: str                            # plain English, for the landing page
    family: str                          # one of FAMILIES
    one_line: str                        # what the motion is, in one sentence
    use_cases: tuple[str, ...]           # from USE_CASE_TAGS
    space: str                           # one of SPACES (``defined_in`` is this field)
    starts_as: str                       # one of STARTS_AS
    scope: str                           # one of SCOPES
    output: str                          # one of OUTPUTS
    needs: tuple[str, ...]               # from NEEDS
    parameters: tuple[Parameter, ...]    # (name, unit, default), SI
    segments_join: str                   # one of JOINS
    exists_in: tuple[str, ...]           # "path:line"; a path relative to the bam_ws root, or absolute for another repo
    prior_art: tuple[Citation, ...]      # (citation, URL) primary sources
    priority: str                        # one of PRIORITIES
    notes: str = ""
    dimensions: tuple[str, ...] = ()     # from DIMENSIONS; authored for Cartesian kinds only, derived from scope for the rest

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("_", "").isalnum() or self.id != self.id.lower():
            raise ValueError(f"id {self.id!r} is not snake_case")
        for field_name, value, vocabulary in (("family", self.family, FAMILIES), ("space", self.space, SPACES),
                                              ("starts_as", self.starts_as, STARTS_AS),
                                              ("scope", self.scope, SCOPES), ("output", self.output, OUTPUTS),
                                              ("segments_join", self.segments_join, JOINS), ("priority", self.priority, PRIORITIES)):
            _refuse_unknown(field_name, (value,), vocabulary, self.id)
        _refuse_unknown("use_cases", self.use_cases, USE_CASE_TAGS, self.id)
        _refuse_unknown("needs", self.needs, NEEDS, self.id)
        if self.space == "cartesian":
            if not self.dimensions:
                raise ValueError(f"{self.id}: a Cartesian kind names its geometry's dimensions (line 1D, plane 2D, volume 3D, + orientation 6D)")
        elif self.dimensions:
            raise ValueError(f"{self.id}: a {self.space}-space kind's dimensions follow from its scope; do not author them")
        else:
            # WHY object.__setattr__: the dataclass is frozen; this fills the derived default once, at construction.
            object.__setattr__(self, "dimensions", DIMENSIONS_OF_SCOPE[self.scope])
        _refuse_unknown("dimensions", self.dimensions, DIMENSIONS, self.id)
        positions = [DIMENSIONS.index(value) for value in self.dimensions]
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise ValueError(f"{self.id}: dimensions {self.dimensions} are not a contiguous run of {DIMENSIONS}")
        for parameter in self.parameters:
            if len(parameter) != 3 or not isinstance(parameter[0], str) or not isinstance(parameter[1], str):
                raise ValueError(f"{self.id}: parameter {parameter!r} is not (name, unit, default)")
        for citation in self.prior_art:
            if len(citation) != 2 or not str(citation[1]).startswith("http"):
                raise ValueError(f"{self.id}: citation {citation!r} is not (citation, URL)")

    @property
    def joint_trajectory(self) -> bool:
        """True when the kind is a position JointTrajectory the JointTrajectoryController can play as is."""

        return self.output == "joint_trajectory" and self.space != "torque"

    @property
    def defined_in(self) -> str:
        """The space the kind's geometry or signal is written in: ``joint``, ``cartesian`` or ``torque`` (the ``space`` field)."""

        return self.space

    @property
    def entry_level(self) -> str:
        """The lowest rung of ENTRY_LEVELS whose inputs generate this kind: the highest rung any of its ``needs`` sits on."""

        rungs = [ENTRY_LEVEL_OF_NEED[need] for need in self.needs]
        return max((rung for rung in rungs if rung), key=ENTRY_LEVELS.index, default=ENTRY_LEVELS[0])

    @property
    def entry_extras(self) -> tuple[str, ...]:
        """What the kind needs beside the ladder (``collision``), in ENTRY_EXTRAS order."""

        extras = {ENTRY_EXTRA_OF_NEED[need] for need in self.needs if need in ENTRY_EXTRA_OF_NEED}
        return tuple(extra for extra in ENTRY_EXTRAS if extra in extras)
