"""Lane C's own fixtures (numpy only; stand-ins, labelled): a dynamics model for the five-bar and the UR5e (``input_shaped_move``
needs the rung), and a lightly damped mode to hold a shaped move against."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiagonalDynamics:
    """STAND-IN: a per-joint inertia, viscous damping and a constant gravity torque: tau = J qdd + b qd + g. Made-up values for
    two- and six-joint tests of kinds that need the dynamics rung; not any robot's identified model."""

    inertia: tuple[float, ...]
    damping: tuple[float, ...]
    gravity: tuple[float, ...]
    name: str = "diagonal stand-in (J qdd + b qd + g)"
    cache_key: str = "lane-c-diagonal-1"

    def torque(self, q, qd, qdd):
        qd, qdd = np.asarray(qd, float), np.asarray(qdd, float)
        return np.asarray(self.inertia) * qdd + np.asarray(self.damping) * qd + np.asarray(self.gravity)


def diagonal_dynamics(dof: int) -> DiagonalDynamics:
    return DiagonalDynamics((0.5,) * dof, (0.05,) * dof, (0.1,) * dof)


def residual_vibration(t: np.ndarray, reference: np.ndarray, mode_hz: float, damping_ratio: float, *, after_s: float) -> float:
    """The vibration (rad) left in a lightly damped mode, a second-order low-pass of the reference (damped frequency ``mode_hz``,
    damping ``damping_ratio``), after the reference has stopped: the peak |response - final reference| over t >= ``after_s``.
    Integrated exactly per sample (a zero-order hold of the reference), no library."""
    wd = 2.0 * np.pi * mode_hz
    wn = wd / np.sqrt(1.0 - damping_ratio**2)
    zeta = damping_ratio
    dt = float(t[1] - t[0])
    # state-space x'' + 2 zeta wn x' + wn^2 x = wn^2 r, exact discretisation by the matrix exponential's closed form (2x2)
    a = np.array([[0.0, 1.0], [-wn**2, -2.0 * zeta * wn]])
    b = np.array([0.0, wn**2])
    eigenvalues, vectors = np.linalg.eig(a)
    phi = (vectors @ np.diag(np.exp(eigenvalues * dt)) @ np.linalg.inv(vectors)).real
    gamma = (np.linalg.inv(a) @ (phi - np.eye(2)) @ b)
    state = np.array([reference[0], 0.0])
    out = np.empty(len(t))
    for index, value in enumerate(reference):
        out[index] = state[0]
        state = phi @ state + gamma * value
    late = t >= after_s
    return float(np.max(np.abs(out[late] - reference[-1])))
