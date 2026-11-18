from __future__ import annotations

from .. import kernels as k

from ..kind import Context, Kind, param


class Multisine(Kind):
    """Schroeder-phased sines on every joint at once, each joint on its own lines: coupled dynamics in one run."""

    catalog_id = "multisine"
    family = "excitation"
    use_cases = ("full_dynamics_id", "bandwidth")

    joints: tuple[str, ...] | None = param(None, "joints", "the joints that move (default: every joint)")
    band_hz: tuple[float, float] = param((0.2, 3.0), "Hz", "the lowest and highest excited frequency")
    period_s: float = param(10.0, "s", "one period; the frequency resolution is 1 / period_s")
    periods: int = param(2, "count", "whole periods to play")
    seed: int = param(0, "seed", "the phase offsets' seed")
    fade_s: float = param(1.0, "s", "raised-cosine fade in and out")
    margin: float = param(0.8, "1", "the tightest of room, velocity and acceleration the amplitude reaches")

    def pieces(self, context: Context) -> list[k.Arrays]:
        return [k.multisine(context.start, context.joints(self.joints), context.room_around(context.start), context.rate_hz,
                            band_hz=self.band_hz, period_s=self.period_s, periods=self.periods, seed=self.seed,
                            fade_s=self.fade_s, margin=self.margin, label=self.display_label)]
