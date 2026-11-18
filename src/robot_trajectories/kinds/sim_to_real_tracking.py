from __future__ import annotations

from ..registry import kind
from ..segments import annotate, chain
from .deployment_motion_classes import HELD_OUT_SEEDS, deployment_motion_classes
from .hold import hold
from .linear_chirp import linear_chirp
from .repeat_floor_tape import repeat_floor_tape
from .speed_ladder import speed_ladder
from .step_train import step_train
from .velocity_plateaus import velocity_plateaus


@kind(use_cases=("tracking_eval", "repeatability", "speed_limits", "stop_accuracy", "pid_tuning", "friction_id", "inertia_id"),
      family="standard_test", starts_as="path", dimensions=("1D",), higher_order=True)
def sim_to_real_tracking(space, joint=None, *, campaign_round=0, start=None):
    """Does the sim track the real joint, at rising speed, above its repeat floor? The pendulum campaign's tapes in the order a
    session runs them: read the floor first (every gate is a multiple of it), identify the twin, check the PID, then the
    speed ladder and the motion classes, the held-out seeds last. Each segment says to wait after it (one tape at a time)."""
    def step(trajectory, purpose):
        return annotate(trajectory, purpose, wait_after=True)

    settle = step(hold(space, 2.0, label="settle", start=start), "at rest before anything moves: the stop error with no motion behind it")
    floor = step(repeat_floor_tape(space, joint, start=settle),
                 "one tape five times, the same input: the run-to-run floor (sim vs real must stay within 1.5x it)")
    sweep = step(linear_chirp(space, joint, start=floor), "0.5-2 Hz at a constant speed amplitude: inertia and delay for the twin")
    plateaus = step(velocity_plateaus(space, joint, start=sweep), "constant-speed plateaus out and back: friction against speed")
    steps = step(step_train(space, joint, start=plateaus),
                 "steps into holds, two large ones across the friction band: overshoot, ringing, settling")
    ladder = step(speed_ladder(space, joint, campaign_round=campaign_round, start=steps),
                  "A-to-B at 20-100 % of top speed, 5 repeats a rung + 2 held-out pairs: tracking and the 1 mm stop as speed rises")
    classes = step(deployment_motion_classes(space, joint, label="deployment classes", start=ladder),
                   "pickplace, cycle, jog, sweep on seeds 0-2: what candidate gains are scored on")
    held_out = step(deployment_motion_classes(space, joint, seeds=HELD_OUT_SEEDS, label="deployment classes, held out", start=classes),
                    "the same classes on the sealed seeds 3-5: scored, never used to pick")
    return chain(settle, floor, sweep, plateaus, steps, ladder, classes, held_out)
