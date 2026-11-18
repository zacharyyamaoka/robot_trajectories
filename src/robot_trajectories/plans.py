"""What do you want to test? One plan per purpose (V3's front door), robot-free: each returns a Plan of records.

    plans.sim_to_real_tracking(joint="can11")   # this week on the pendulum: floor, identification, steps, ladder, classes
    plans.pid_tuning(joint="can11")             # step train, settle, chirp, settle
    plans.friction_id(joint="can11")            # velocity sweep
    plans.tracking_check(count=8, seed=1)       # random moves, home
    plans.workspace_survey()                    # reachable tour (a box grid without a reach map), home
    plans.dynamics_excitation()                 # multisine on every joint (+ a pose grid when asked)

    plans.for_use_case("pid_tuning")            # the plans that serve it
    generator.generate(plans.pid_tuning(joint="can11") + plans.friction_id(joint="can11"))

WHY all plans in this one file (V3): it is the menu. Reading it top to bottom answers "what do I run for X, and what will
it do?". A plan is a few lines of records with purposes, so the 61-kind catalog never has to be hand-kept here: kinds
live one per file under kinds/, and a plan only names the ones a purpose needs.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .kinds import (Chirp, DeploymentMotionClasses, Hold, Home, LinearChirp, Move, Multisine, RandomWaypoints, Repeat,
                    SpeedLadder, StepTrain, VelocityPlateaus, VelocitySweep, WorkspaceSweep)
from .plan import Plan, PlanStep, step
from .time_law import TimeLaw


@dataclass(frozen=True)
class PlanInfo:
    """A plan as the catalog lists it: the question it answers and the use cases it serves."""

    name: str
    use_cases: tuple[str, ...]
    question: str
    make: Callable[..., Plan]


_PLANS: dict[str, PlanInfo] = {}


def plan(*, use_cases: Sequence[str], question: str):
    """Make ``function(...) -> [step(...), ...]`` a plan: it returns a Plan named for the function, its last step
    ``then="wait"`` (a plan is one test; its end is where you look), and it is listed under its use cases."""

    def decorate(function: Callable[..., list[PlanStep]]) -> Callable[..., Plan]:
        name = function.__name__

        @functools.wraps(function)
        def make(*args: Any, **options: Any) -> Plan:
            steps = [replace(item, plan=name) for item in function(*args, **options)]
            steps[-1] = replace(steps[-1], then="wait")
            parts = [repr(arg) for arg in args] + [f"{key}={value!r}" for key, value in options.items()]
            return Plan(name, tuple(steps), (f"plans.{name}({', '.join(parts)})",))

        _PLANS[name] = PlanInfo(name, tuple(use_cases), question, make)
        return make

    return decorate


# ====================================================================== this week on the pendulum


@plan(use_cases=("tracking_eval", "repeatability", "speed_limits", "stop_accuracy", "pid_tuning", "friction_id", "inertia_id"),
      question="Does the sim track the real joint, at rising speed, above its repeat floor?")
def sim_to_real_tracking(joint: str | None = None, *, round: int = 0) -> list[PlanStep]:
    """The pendulum campaign's tapes in the order a session runs them: read the floor first (every gate is a multiple of
    it), identify the twin, check the PID, then the speed ladder and the deployment classes, the held-out seeds last."""
    floor_tape = Move(joint=joint, targets_fraction=(0.7, -0.7), time_law=TimeLaw.minimum_jerk(peak_speed_fraction=0.6),
                      dwell_s=1.0, label="floor tape")
    classes = DeploymentMotionClasses(joint=joint, label="deployment classes")
    return [
        step(Hold(duration_s=2.0, label="settle"), "at rest before anything moves: the stop error with no motion behind it",
             then="wait"),
        step(Repeat(records=(floor_tape,), times=5, label="repeat floor"),
             "one tape five times, the same input: the run-to-run floor (sim vs real must stay within 1.5x it)", then="wait"),
        step(LinearChirp(joint=joint), "0.5-2 Hz at a constant speed amplitude: inertia and delay for the twin", then="wait"),
        step(VelocityPlateaus(joint=joint), "constant-speed plateaus out and back: friction against speed", then="wait"),
        step(StepTrain(joint=joint), "steps into holds, two large ones across the friction band: overshoot, ringing, settling",
             then="wait"),
        step(SpeedLadder(joint=joint, round=round), "A-to-B at 20-100 % of top speed, 5 repeats a rung + 2 held-out pairs: "
             "tracking and the 1 mm stop as speed rises", then="wait"),
        step(classes, "pickplace, cycle, jog, sweep on seeds 0-2: what candidate gains are scored on", then="wait"),
        step(classes.held_out(), "the same classes on the sealed seeds 3-5: scored, never used to pick"),
    ]


# ====================================================================== the purposes (the round's plans, V3)


@plan(use_cases=("pid_tuning", "bandwidth"), question="Is this joint's PID right?")
def pid_tuning(joint: str | None = None, *, steps_fraction: Sequence[float] | None = None, dwell_s: float | None = None,
               chirp_hz: tuple[float, float] = (0.2, 5.0), chirp_s: float = 20.0, settle_s: float = 2.0,
               end_settle_s: float = 1.0) -> list[PlanStep]:
    train = StepTrain(joint=joint)
    if steps_fraction is not None:
        train = replace(train, steps_fraction=tuple(steps_fraction))
    if dwell_s is not None:
        train = replace(train, dwell_s=dwell_s)
    return [
        step(train, "step response: overshoot, rise and settling time, steady-state error"),
        step(Hold(duration_s=settle_s), "settle, so the log separates the step response from the chirp"),
        step(Chirp(joint=joint, start_hz=chirp_hz[0], end_hz=chirp_hz[1], duration_s=chirp_s),
             f"frequency response, {chirp_hz[0]:g}-{chirp_hz[1]:g} Hz: bandwidth, phase margin, resonances"),
        step(Hold(duration_s=end_settle_s), "settle before the next test"),
    ]


@plan(use_cases=("friction_id",), question="How much friction does this joint have?")
def friction_id(joint: str | None = None, *, speeds_fraction: Sequence[float] = (0.1, 0.3, 0.6)) -> list[PlanStep]:
    speeds = ", ".join(f"{100 * fraction:.0f}" for fraction in speeds_fraction)
    return [step(VelocitySweep(joint=joint, speeds_fraction=tuple(speeds_fraction)),
                 f"constant-speed passes at {speeds} % of top speed, up and down: Coulomb and viscous friction, Stribeck at the slow end")]


@plan(use_cases=("tracking_eval", "smoke_test"), question="Does it follow ordinary moves?")
def tracking_check(*, count: int = 8, seed: int = 0) -> list[PlanStep]:
    return [step(RandomWaypoints(count=count, seed=seed), f"{count} seeded random moves (seed {seed}): tracking error, RMS and peak"),
            step(Home(), "back home, so the next plan begins where this one did")]


@plan(use_cases=("workspace_coverage", "accuracy", "kinematic_calibration"), question="Where can it go, and how accurately?")
def workspace_survey(*, grid_deg: float = 12.0, joints: Sequence[str] | None = None, points_per_joint: int = 3,
                     speed: float = 0.8, dwell_s: float = 0.05) -> list[PlanStep]:
    sweep = WorkspaceSweep(grid_step=math.radians(grid_deg), joints=None if joints is None else tuple(joints),
                           points_per_joint=points_per_joint, speed=speed, dwell_s=dwell_s)
    return [step(sweep, f"every reachable configuration on a {grid_deg:g}° grid, neighbour moves only (a "
                        f"{points_per_joint}-per-joint box grid without a reach map): coverage, accuracy, calibration poses"),
            step(Home(), "back home")]


@plan(use_cases=("full_dynamics_id", "inertia_id"), question="What are its inertia, gravity and coupling?")
def dynamics_excitation(*, joints: Sequence[str] | None = None, band_hz: tuple[float, float] = (0.2, 3.0), period_s: float = 10.0,
                        periods: int = 2, seed: int = 0, pose_grid_joints: Sequence[str] | None = None,
                        points_per_joint: int = 3) -> list[PlanStep]:
    """``pose_grid_joints`` adds a grid of static poses over those joints (an arm's gravity terms)."""
    steps = [step(Multisine(joints=None if joints is None else tuple(joints), band_hz=tuple(band_hz), period_s=period_s,
                            periods=periods, seed=seed),
                  f"{band_hz[0]:g}-{band_hz[1]:g} Hz multisine on every joint at once: inertia, coupling, damping")]
    if pose_grid_joints:
        # WHY static poses on an arm (V3): gravity loads change with configuration; a multisine about one pose barely sees them.
        steps += [step(WorkspaceSweep(joints=tuple(pose_grid_joints), points_per_joint=points_per_joint, speed=0.6, dwell_s=0.2,
                                      label="pose grid"),
                       f"{points_per_joint ** len(pose_grid_joints)} poses over {', '.join(pose_grid_joints)}: the gravity terms, at rest"),
                  step(Home(), "back home")]
    return steps


# ====================================================================== the menu


def catalog() -> tuple[PlanInfo, ...]:
    return tuple(_PLANS.values())


def get(name: str) -> PlanInfo:
    try:
        return _PLANS[name]
    except KeyError:
        raise KeyError(f"no plan {name!r}; plans are {tuple(_PLANS)}") from None


def for_use_case(use_case: str) -> tuple[str, ...]:
    return tuple(info.name for info in _PLANS.values() if use_case in info.use_cases)


def custom(name: str, steps: Sequence[PlanStep]) -> Plan:
    """Your own plan from records: ``plans.custom("two tones", [step(SineDwell(frequency_hz=1.0), "gain at 1 Hz"), ...])``."""
    steps = [replace(item, plan=name) for item in steps]
    return Plan(name, tuple(steps), (f"plans.custom({name!r}, [{len(steps)} steps])",))


def from_json(text: str) -> Plan:
    return Plan.from_json(text)
