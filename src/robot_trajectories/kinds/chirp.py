from __future__ import annotations

from .. import kernels as k

from ..kind import Context, Kind, param


class Chirp(Kind):
    """A sine swept from start_hz to end_hz on one joint at ONE amplitude for the whole sweep (the shared kernel): its
    frequency response and bandwidth.

    WHY it keeps the shared kernel, whose highest frequency sets the amplitude everywhere: the round's frozen scenario is
    this motion, sample for sample. ``LinearChirp`` carries the catalog's constant-speed rule (larger at low frequency)."""

    catalog_id = "log_chirp"
    family = "excitation"
    use_cases = ("pid_tuning", "bandwidth", "inertia_id")  # the round's tags (catalog.json's log_chirp omits pid_tuning)

    joint: str | int | None = param(None, "joint", "the joint that moves (name or index; optional on a 1-DOF robot)")
    start_hz: float = param(0.2, "Hz", "the sweep's first frequency")
    end_hz: float = param(5.0, "Hz", "the sweep's last frequency")
    duration_s: float = param(20.0, "s", "the sweep's length, fades included")
    logarithmic: bool = param(True, "bool", "equal time per octave (True) or per hertz (False)")
    fade_s: float = param(1.0, "s", "raised-cosine fade in and out, so it starts and ends at rest")
    margin: float = param(0.8, "1", "the tightest of room, velocity and acceleration the amplitude reaches")

    def check(self) -> None:
        if self.start_hz <= 0 or self.end_hz <= 0:
            raise ValueError(f"Chirp frequencies must be positive, got {self.start_hz} and {self.end_hz} Hz")

    def pieces(self, context: Context) -> list[k.Arrays]:
        return [k.chirp(context.start, context.joint(self.joint), context.room_around(context.start), context.rate_hz,
                        start_hz=self.start_hz, end_hz=self.end_hz, duration_s=self.duration_s, logarithmic=self.logarithmic,
                        fade_s=self.fade_s, margin=self.margin, label=self.display_label)]
