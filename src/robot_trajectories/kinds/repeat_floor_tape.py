from __future__ import annotations

from ..registry import kind
from ..segments import repeat
from ..time_law import TimeLaw
from .move import move


@kind(use_cases=("repeatability", "tracking_eval"), catalog_id="repeat_floor_tape", higher_order=True)
def repeat_floor_tape(space, joint=None, *, targets_fraction=(0.7, -0.7), time_law=TimeLaw.minimum_jerk(peak_speed_fraction=0.6),
                      dwell_s=1.0, times=5, label="repeat floor", start=None):
    """One frozen tape played ``times`` times with the same input: the run-to-run floor every sim-vs-real gate is a multiple of
    (the pendulum contract: sim vs real <= 1.5 x the same-input repeat floor, 5 repeats).

    The tape is a ``move`` through ``targets_fraction`` of the room and back, under ``time_law``, dwelling ``dwell_s`` at each
    stop (the week's floor tape, PEP 0001 section 9). Any other tape: ``repeat(trajectory, times)``."""
    tape = move(space, joint, targets_fraction=targets_fraction, time_law=time_law, dwell_s=dwell_s, label="floor tape", start=start)
    return repeat(tape, times, label=label)
