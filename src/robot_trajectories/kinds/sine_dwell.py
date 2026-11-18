from __future__ import annotations

from .. import kernels as k

from ..kind import Context, Kind, param


class SineDwell(Kind):
    """One sine at a fixed frequency on one joint, faded in and out: the gain and phase at that one frequency."""

    catalog_id = "faded_sine"
    family = "excitation"
    use_cases = ("bandwidth",)

    joint: str | int | None = param(None, "joint", "the joint that moves (name or index; optional on a 1-DOF robot)")
    frequency_hz: float = param(1.0, "Hz", "the one frequency")
    duration_s: float = param(10.0, "s", "how long it dwells there, fades included")
    fade_s: float = param(1.0, "s", "raised-cosine fade in and out")
    margin: float = param(0.8, "1", "the tightest of room, velocity and acceleration the amplitude reaches")

    def pieces(self, context: Context) -> list[k.Arrays]:
        return [k.chirp(context.start, context.joint(self.joint), context.room_around(context.start), context.rate_hz,
                        start_hz=self.frequency_hz, end_hz=self.frequency_hz, duration_s=self.duration_s, logarithmic=False,
                        fade_s=self.fade_s, margin=self.margin, label=self.display_label)]
