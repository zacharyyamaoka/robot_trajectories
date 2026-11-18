"""Lane A's own fixtures (numpy only; every object here is a labelled stand-in, never a robot's identified model)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArmDynamics:
    """A made-up J-joint arm that is LINEAR in its parameters, so its regressor is exact: for joint i

        tau_i = I_i qdd_i + sum_j k_ij cos(q_i - q_j) qdd_j + b_i qd_i + f_i tanh(qd_i / 0.05) + gs_i sin(q_i) + gc_i cos(q_i)

    with a coupling k_ij between every pair. ``regressor(q, qd, qdd)`` is the (J, P) matrix Y with ``torque = Y @ parameters``
    (P = 5 J + J (J - 1) / 2). Stand-in for lane A's fourier_series_excitation test: not the five-bar's or the UR5e's model."""

    dof: int
    name: str = "lane-A stand-in arm (linear in its parameters)"
    cache_key: str = "lane-a-arm-1"

    @property
    def parameters(self) -> np.ndarray:
        own = np.concatenate([np.linspace(0.8, 0.3, self.dof), np.linspace(0.4, 0.1, self.dof), np.linspace(0.3, 0.1, self.dof),
                              np.linspace(2.0, 0.5, self.dof), np.linspace(0.5, 0.2, self.dof)])
        return np.concatenate([own, np.full(self.dof * (self.dof - 1) // 2, 0.05)])

    def regressor(self, q, qd, qdd) -> np.ndarray:
        q, qd, qdd = (np.asarray(value, float) for value in (q, qd, qdd))
        n = self.dof
        rows = np.zeros((n, 5 * n + n * (n - 1) // 2))
        for i in range(n):
            rows[i, i] = qdd[i]
            rows[i, n + i] = qd[i]
            rows[i, 2 * n + i] = np.tanh(qd[i] / 0.05)
            rows[i, 3 * n + i] = np.sin(q[i])
            rows[i, 4 * n + i] = np.cos(q[i])
        pair = 5 * n
        for i in range(n):
            for j in range(i + 1, n):
                rows[i, pair] = np.cos(q[i] - q[j]) * qdd[j]
                rows[j, pair] = np.cos(q[i] - q[j]) * qdd[i]
                pair += 1
        return rows

    def torque(self, q, qd, qdd) -> np.ndarray:
        return self.regressor(q, qd, qdd) @ self.parameters


@dataclass(frozen=True)
class TorqueOnlyDynamics:
    """The same arm with only ``torque`` (no ``regressor``): the model a caller who cannot give a regressor would inject."""

    inner: ArmDynamics
    name: str = "lane-A stand-in arm, torque only"
    cache_key: str = "lane-a-arm-torque-only-1"

    def torque(self, q, qd, qdd) -> np.ndarray:
        return self.inner.torque(q, qd, qdd)


def with_named_home(space, home=None):
    """The same space with its home named (the catalog's ``home_pose`` need); ``home`` defaults to the space's own."""
    return dataclasses.replace(space, home=np.asarray(space.home if home is None else home, float), named_home=True)
