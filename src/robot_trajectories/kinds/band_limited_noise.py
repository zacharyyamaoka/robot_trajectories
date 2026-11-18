from __future__ import annotations

import numpy as np

from .. import kernels as k
from ..registry import kind
from ..segments import Motion
from ._lane_a import deg, joint_room, limit_gain, single_joint


@kind(use_cases=("bandwidth", "inertia_id", "state_coverage"), catalog_id="band_limited_noise")
def band_limited_noise(space, joint=None, *, band_low_hz=0.1, band_high_hz=4.0, rms_rad=0.05, duration_s=30.0, seed=0, fade_s=1.0,
                       margin=0.8, label="band-limited noise", start=None):
    """Gaussian noise filtered to a band in the frequency domain (everything outside it set to zero), faded in and out so it starts
    and ends at rest, scaled so speed and acceleration stay inside the limits: a new realisation per ``seed`` (idinput's 'rgs').
    It needs windowing or averaging across realisations, where a periodic multisine needs neither.

    ``rms_rad``: the asked rms of the unfaded noise; the limits can lower it (``margin`` of the tightest of room, velocity and
    acceleration), and a note says by how much.

    WHY the velocity and acceleration are the spectrum's own derivatives and not a differenced position: the band-limited signal
    is periodic on the run's length, so multiplying its spectrum by i w and -w^2 gives them exactly, and the fade's product rule
    then keeps them consistent with the position to the last sample."""
    if not 0 < band_low_hz < band_high_hz or rms_rad <= 0 or duration_s <= 0:
        raise ValueError("band_limited_noise needs 0 < band_low_hz < band_high_hz, rms_rad > 0 and duration_s > 0")
    if band_high_hz >= 0.5 * space.rate_hz:
        raise ValueError(f"band_high_hz {band_high_hz:g} is not below the Nyquist frequency {0.5 * space.rate_hz:g} Hz of the {space.rate_hz:g} Hz clock")
    index = space.joint_index(joint)
    t = k._samples(duration_s, space.rate_hz)
    count = len(t)
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(count))
    frequencies = np.fft.rfftfreq(count, 1.0 / space.rate_hz)
    spectrum = spectrum * ((frequencies >= band_low_hz) & (frequencies <= band_high_hz))
    omega = 2 * np.pi * frequencies
    x = np.fft.irfft(spectrum, count)
    scale = rms_rad / float(np.std(x))
    x, dx, ddx = (scale * np.fft.irfft(factor * spectrum, count) for factor in (1.0, 1j * omega, -(omega**2)))
    w, dw, ddw = k._fade(t, t[-1], fade_s)
    x, dx, ddx = x * w, dx * w + x * dw, ddx * w + 2 * dx * dw + x * ddw
    gain = min(1.0, limit_gain(joint_room(start, space.room(start), index), space.room(start), index, x, dx, ddx, margin))
    note = f"{label}: {deg(gain * rms_rad)} rms in {band_low_hz:g}-{band_high_hz:g} Hz, seed {seed}"
    if gain < 1.0:
        note += f" (asked {deg(rms_rad)}: x{gain:.3f} so the tightest of room, velocity and acceleration is at {margin:g} of its limit)"
    return Motion([single_joint(t, start, index, gain * x, gain * dx, gain * ddx, label)], [note])
