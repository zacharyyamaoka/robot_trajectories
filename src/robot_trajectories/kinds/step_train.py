from __future__ import annotations

from .. import kernels as k
from ..registry import kind

# The catalog row's defaults (PID tuna's tapes.steps): the first and last 15 % of the room, four small ones between at the size
# PID tuna's sharp step reaches on CAN 11 (top speed x 2 command periods / 1.875 = 1.2 deg = 6.5 % of the room), 0.8 s holds.
STEPS_FRACTION = (0.15, -0.065, 0.065, -0.065, 0.065, -0.15)
DWELL_S = 0.8


@kind(use_cases=("pid_tuning", "tracking_eval", "bandwidth", "stop_accuracy"), catalog_id="step_train")
def step_train(space, joint=None, *, steps_fraction=STEPS_FRACTION, dwell_s=DWELL_S, speed=1.0, label="step train", start=None):
    """The fastest steps the limits allow, each into a hold, alternating sign, with two larger steps that cross the friction
    band: overshoot, ringing and settling are read from it. Each step's target is a fraction of the room around the start;
    it ends back there.

    WHY not PID tuna's 20 ms "sharp" move itself: on the pendulum's 7.39 rad/s^2 it would need 40x the acceleration limit;
    the sharpest step inside the limits is a minimum-jerk move at full speed, which is what this kind generates."""
    if any(abs(fraction) > 1.0 for fraction in steps_fraction):
        raise ValueError(f"step_train.steps_fraction must lie in [-1, 1], got {steps_fraction}")
    return k.step_train(start, space.joint_index(joint), space.room(start), space.rate_hz, steps_fraction=steps_fraction,
                        dwell_s=dwell_s, speed=speed, label=label)
