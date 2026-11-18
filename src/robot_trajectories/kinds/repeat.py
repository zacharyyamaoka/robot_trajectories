from __future__ import annotations

import numpy as np

from .. import kernels as k

from ..kind import Context, Kind, param


class Repeat(Kind):
    """One frozen tape played N times with the same input: the run-to-run floor every sim-vs-real gate is a multiple of
    (the pendulum contract: sim vs real <= 1.5 x the same-input repeat floor, 5 repeats).

    ``records`` is the tape (any records, generated in turn); every repeat starts from the same place at rest, so the
    input really is the same: if the tape ends elsewhere, a move back to its start comes before the next repeat."""

    catalog_id = "repeat_floor_tape"
    family = "standard_test"
    use_cases = ("repeatability", "tracking_eval")
    needs = ("joint_limits", "velocity_limits", "acceleration_limits")

    records: tuple[Kind, ...] = param((), "records", "the tape: the records one repeat plays")
    times: int = param(5, "count", "how many repeats (the catalog's repeats)")

    def check(self) -> None:
        if not self.records or not all(isinstance(record, Kind) for record in self.records):
            raise ValueError("Repeat needs records=(<record>, ...): the tape one repeat plays")
        if self.times < 1:
            raise ValueError(f"Repeat.times must be at least 1, got {self.times}")

    def pieces(self, context: Context) -> list[k.Arrays]:
        start = context.start
        here = start
        pieces: list[k.Arrays] = []
        for repeat in range(1, self.times + 1):
            prefix = f"{self.display_label}: {repeat}/{self.times}"
            if not np.allclose(here, start, atol=1e-9):
                pieces.append(k.min_jerk(here, start, context.bounds, context.rate_hz, speed=0.5, label=f"{prefix}: back to start"))
                here = start
            for record in self.records:
                inner = record.pieces(context.at(here))
                if inner and not np.allclose(inner[0].q[0], here, atol=1e-9):
                    inner.insert(0, k.min_jerk(here, inner[0].q[0], context.bounds, context.rate_hz, speed=0.5, label="transition"))
                pieces += [k.Arrays(piece.t, piece.q, piece.qd, piece.qdd, f"{prefix}: {piece.label}") for piece in inner]
                if inner:
                    here = inner[-1].q[-1]
        return pieces
