from __future__ import annotations

from .. import kernels as k

from ..kind import Context, Kind, param


class StepTrain(Kind):
    """The fastest steps the limits allow, each into a hold, alternating sign, with two larger steps that cross the
    friction band: overshoot, ringing and settling are read from it.

    Defaults follow the catalog row (PID tuna's ``tapes.steps``): count 6, the first and last 15 % of the room, four
    small ones between at the size PID tuna's sharp step reaches on CAN 11 (top speed x 2 command periods / 1.875 =
    1.2 deg = 6.5 % of the room), 0.8 s holds. WHY not PID tuna's 20 ms "sharp" move itself: on the pendulum's
    7.39 rad/s^2 it would need 40x the acceleration limit; the sharpest step inside the limits is a minimum-jerk move at
    full speed, which is what this kind generates."""

    catalog_id = "step_train"
    family = "controller_test"
    use_cases = ("pid_tuning", "tracking_eval", "bandwidth", "stop_accuracy")
    needs = ("joint_limits", "velocity_limits", "acceleration_limits")

    joint: str | int | None = param(None, "joint", "the joint that steps (name or index; optional on a 1-DOF robot)")
    steps_fraction: tuple[float, ...] = param((0.15, -0.065, 0.065, -0.065, 0.065, -0.15), "1",
                                              "each step's target as a fraction of the room around the start; it ends back there")
    dwell_s: float = param(0.8, "s", "the hold after each step (the catalog's hold_s)")
    speed: float = param(1.0, "1", "fraction of the velocity and acceleration limits a step may use")

    def check(self) -> None:
        if any(abs(fraction) > 1.0 for fraction in self.steps_fraction):
            raise ValueError(f"StepTrain.steps_fraction must lie in [-1, 1], got {self.steps_fraction}")

    def pieces(self, context: Context) -> list[k.Arrays]:
        return k.step_train(context.start, context.joint(self.joint), context.room_around(context.start), context.rate_hz,
                            steps_fraction=self.steps_fraction, dwell_s=self.dwell_s, speed=self.speed, label=self.display_label)
