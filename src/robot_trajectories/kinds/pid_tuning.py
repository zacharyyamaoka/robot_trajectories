from __future__ import annotations

from ..registry import kind
from ..segments import annotate, chain
from .chirp import chirp
from .hold import hold
from .step_train import DWELL_S, STEPS_FRACTION, step_train


@kind(use_cases=("pid_tuning", "bandwidth"), family="controller_test", starts_as="time_signal", dimensions=("1D",), higher_order=True)
def pid_tuning(space, joint=None, *, steps_fraction=STEPS_FRACTION, dwell_s=DWELL_S, chirp_hz=(0.2, 5.0), chirp_s=20.0, settle_s=2.0,
               end_settle_s=1.0, start=None):
    """Is this joint's PID right? A step train (overshoot, rise and settling time), a settle, a chirp over ``chirp_hz``
    (bandwidth, phase margin, resonances), a settle. A higher-order kind: four lower kinds chained, recorded as this one call."""
    train = annotate(step_train(space, joint, steps_fraction=steps_fraction, dwell_s=dwell_s, start=start),
                     "step response: overshoot, rise and settling time, steady-state error")
    settle = annotate(hold(space, settle_s, start=train), "settle, so the log separates the step response from the chirp")
    sweep = annotate(chirp(space, joint, start_hz=chirp_hz[0], end_hz=chirp_hz[1], duration_s=chirp_s, start=settle),
                     f"frequency response, {chirp_hz[0]:g}-{chirp_hz[1]:g} Hz: bandwidth, phase margin, resonances")
    rest = annotate(hold(space, end_settle_s, start=sweep), "settle before the next test", wait_after=True)
    return chain(train, settle, sweep, rest)
