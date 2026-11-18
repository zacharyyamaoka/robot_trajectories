from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..registry import kind
from ..segments import annotate, chain
from .hold import hold
from .sine_dwell import sine_dwell

RULES = ("constant_speed", "constant_angle")


@kind(use_cases=("bandwidth", "pid_tuning", "delay_id"), catalog_id="stepped_sine_dwell", higher_order=True)
def stepped_sine_dwell(space, joint=None, *, f_low_hz=0.3, f_high_hz=8.0, frequencies=12, settle_cycles=2, measure_cycles=4,
                       ramp_cycles=1, amplitude_rule="constant_speed", rest_s=0.5, margin=0.8, label="stepped sine", start=None):
    """A list of frequencies played one after another, each a ``sine_dwell``: ramp in, settle a few cycles, measure amplitude and
    phase over a few more, ramp out, a short rest. The classic point-by-point Bode measurement: the slowest per point of the
    frequency-response kinds and the most robust to nonlinearity. A higher-order kind: ``sine_dwell`` and ``hold`` chained.

    ``frequencies``: that many, equally spaced in log frequency from ``f_low_hz`` to ``f_high_hz``.
    ``amplitude_rule``: ``constant_speed`` (every dwell as large as the limits allow: the angle shrinks as 1/f once the speed or
    acceleration binds) or ``constant_angle`` (every dwell at the angle the highest frequency allows).

    WHY each dwell is as big as the limits allow and not a fixed angle: a Bode point's signal-to-noise is the amplitude, and the
    joint's limits are the ceiling at each frequency; ``constant_angle`` is there for plants whose gain is amplitude-dependent."""
    if amplitude_rule not in RULES:
        raise ValueError(f"stepped_sine_dwell.amplitude_rule must be one of {RULES}, got {amplitude_rule!r}")
    if f_low_hz <= 0 or f_high_hz < f_low_hz or frequencies < 1:
        raise ValueError(f"stepped_sine_dwell needs 0 < f_low_hz <= f_high_hz and frequencies >= 1, got {f_low_hz}, {f_high_hz}, {frequencies}")
    if settle_cycles < 0 or measure_cycles <= 0 or ramp_cycles <= 0:
        raise ValueError("stepped_sine_dwell needs settle_cycles >= 0, measure_cycles > 0 and ramp_cycles > 0")
    index = space.joint_index(joint)
    list_hz = [float(f) for f in np.geomspace(f_low_hz, f_high_hz, int(frequencies))]
    centre = space.start_at(start)
    bounds = space.room(centre)

    def duration(frequency: float) -> float:
        return (settle_cycles + measure_cycles + 2 * ramp_cycles) / frequency

    scale = {frequency: 1.0 for frequency in list_hz}
    if amplitude_rule == "constant_angle":
        # WHY a first pass on the kernel: the angle each frequency allows at margin 1 is read off the signal itself (the fade
        # and the limits interact), then every dwell is trimmed to the smallest one; the gain is linear, so the trim is exact.
        swing = {f: float(np.max(np.abs(k.chirp(centre, index, bounds, space.rate_hz, start_hz=f, end_hz=f, duration_s=duration(f),
                                                logarithmic=False, fade_s=ramp_cycles / f, margin=1.0).q[:, index] - centre[index])))
                 for f in list_hz}
        smallest = min(swing.values())
        scale = {f: smallest / swing[f] for f in list_hz}
    parts: list = []
    here = start
    for number, frequency in enumerate(list_hz):
        dwell = annotate(sine_dwell(space, joint, frequency_hz=frequency, duration_s=duration(frequency), fade_s=ramp_cycles / frequency,
                                    margin=margin * scale[frequency], label=f"{label}: {frequency:.3g} Hz", start=here),
                         f"{frequency:.3g} Hz: {settle_cycles} cycles to settle, then measure gain and phase over {measure_cycles}")
        parts.append(dwell)
        last = number == len(list_hz) - 1
        if rest_s > 0:
            dwell = annotate(hold(space, rest_s, start=dwell, label=f"{label}: rest"), "rest before the next frequency",
                             wait_after=last)
            parts.append(dwell)
        here = parts[-1]
    return chain(*parts)
