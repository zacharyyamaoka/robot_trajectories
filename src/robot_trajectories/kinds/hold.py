from __future__ import annotations

from .. import kernels as k

from ..kind import Context, Kind, param


class Hold(Kind):
    """Hold the last target and read the settled stop from the hold's final window: the segment that joins two moves,
    and the stop-accuracy reading itself."""

    catalog_id = "hold_at_pose"
    family = "hold"
    use_cases = ("stop_accuracy", "pid_tuning", "repeatability", "smoke_test")
    needs = ("joint_limits",)

    duration_s: float = param(1.0, "s", "how long to stay (the catalog's hold_s)")
    settle_window_s: float = param(0.3, "s", "the last stretch of the hold, where the settled stop is read (a band)")

    def check(self) -> None:
        if self.duration_s < 0 or self.settle_window_s < 0:
            raise ValueError(f"Hold needs duration_s and settle_window_s >= 0, got {self.duration_s}, {self.settle_window_s}")

    def pieces(self, context: Context) -> list[k.Arrays]:
        window = min(self.settle_window_s, self.duration_s)
        settle = self.duration_s - window
        # WHY two pieces: the read window shows as its own band, so a scorer reads the stop where the catalog says to.
        if window <= 0 or settle * context.rate_hz < 2:
            return [k.hold(context.start, self.duration_s, context.rate_hz, label=self.display_label)]
        return [k.hold(context.start, settle, context.rate_hz, label=self.display_label),
                k.hold(context.start, window, context.rate_hz, label=f"{self.display_label}: settled-stop window")]
