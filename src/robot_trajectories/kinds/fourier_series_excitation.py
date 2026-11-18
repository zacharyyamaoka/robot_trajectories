from __future__ import annotations

import math

import numpy as np

from .. import kernels as k
from ..registry import LIMITS, kind
from ..segments import Motion

CRITERIA = ("condition_number", "d_optimality")


def _series(coefficients: np.ndarray, harmonics: np.ndarray, omega: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Swevers' finite Fourier series, zero-mean: x = sum a_k / (k w) sin(k w t) - b_k / (k w) cos(k w t), and its derivatives.
    ``coefficients`` is (J, 2 H): the a's then the b's of every joint; returns three (len(t), J) arrays."""
    count = len(harmonics)
    a, b = coefficients[:, :count], coefficients[:, count:]
    w = harmonics * omega
    sines, cosines = np.sin(np.outer(t, w)), np.cos(np.outer(t, w))
    x = sines @ (a / w).T - cosines @ (b / w).T
    dx = cosines @ a.T + sines @ b.T
    ddx = -sines @ (a * w).T + cosines @ (b * w).T
    return x, dx, ddx


def _features(space, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
    """The dynamics regressor Y (J, P) at one state, with tau = Y pi.

    The model's own ``regressor(q, qd, qdd)`` when it has one (pinocchio's computeJointTorqueRegressor, or a fixture's). Otherwise
    a generic basis built from what every model has: for joint i, [qdd_j (J), qd_j (J), sign(qd_i), the model's own gravity
    torque g_i(q)]: inertia and coupling, damping, Coulomb friction, and a gravity gain (4 J columns)."""
    regressor = getattr(space.dynamics, "regressor", None)
    if callable(regressor):
        return np.asarray(regressor(q, qd, qdd), float)
    dof = len(q)
    gravity = space.torque(q, np.zeros(dof), np.zeros(dof))
    rows = np.zeros((dof, 4 * dof))
    for joint in range(dof):
        rows[joint, :dof] = qdd
        rows[joint, dof:2 * dof] = qd
        rows[joint, 2 * dof + joint] = np.sign(qd[joint])
        rows[joint, 3 * dof + joint] = gravity[joint]
    return rows


def _cost(stack: np.ndarray, criterion: str) -> float:
    """Lower is better. Columns are normalised to unit length first (units differ across columns: an inertia column and a gravity
    column), then the log of the condition number, or minus the log-determinant of Y'Y (D-optimality, <= 0 after the scaling)."""
    norms = np.linalg.norm(stack, axis=0)
    keep = norms > 1e-12 * max(float(norms.max()), 1e-300)
    normalised = stack[:, keep] / norms[keep]
    singular = np.linalg.svd(normalised, compute_uv=False)
    if criterion == "condition_number":
        return float(np.log(singular[0] / max(singular[-1], 1e-300)))
    return float(-np.sum(np.log(np.maximum(singular, 1e-300) ** 2)))


def condition_number(stack: np.ndarray) -> float:
    """The column-normalised condition number of a stacked regressor (the number the kind's note reports)."""
    return float(np.exp(_cost(stack, "condition_number")))


@kind(use_cases=("full_dynamics_id", "gravity_id", "inertia_id", "friction_id", "state_coverage"), needs=(*LIMITS, "dynamics_model"),
      unchecked=("collision_model",), catalog_id="fourier_series_excitation")
def fourier_series_excitation(space, *, joints=None, fundamental_hz=0.1, harmonics=5, periods=4, criterion="condition_number",
                              restarts=24, iterations=300, seed=0, fade_s=1.0, margin=0.8, label="fourier series excitation",
                              start=None):
    """Each joint (``joints``, default all) a finite Fourier series of one fundamental, its coefficients OPTIMISED to make the
    dynamics regressor well-conditioned (Swevers 1997; FloBaRoID's recipe) under the position, speed and acceleration limits (and
    the torque limit when the space states one, and no collision when it carries a checker): the coefficients that identify an arm's
    inertia, gravity, friction and coupling best from one run. ``periods`` periods, faded in and out.

    ``criterion``: ``condition_number`` (of the column-normalised regressor) or ``d_optimality`` (the determinant). The search is
    numpy-only and seeded: ``restarts`` random series, then ``iterations`` random perturbations of the best, each candidate scaled
    to use ``margin`` of its tightest limit (an unscaled shape would compare small against large amplitudes). ``restarts=1,
    iterations=0`` is the un-optimised random series (the baseline the optimisation is measured against).

    It NEEDS the space's dynamics (the objective is the regressor): without one it is refused, never an unoptimised series
    handed over. The regressor is the model's ``regressor(q, qd, qdd) -> (J, P)`` when it has one; otherwise a generic basis built
    from ``torque`` (inertia and coupling, damping, Coulomb friction, gravity gain), noted in the segment.

    WHY the fade, not Swevers' q(0) = 0 constraints: the periodic series is what the regressor sees, and the raised-cosine fade is
    the library's one way to start and end at rest; the optimiser sees the series over one period, and the realised run's
    condition number is reported in the note."""
    if criterion not in CRITERIA:
        raise ValueError(f"fourier_series_excitation.criterion must be one of {CRITERIA}, got {criterion!r}")
    if fundamental_hz <= 0 or harmonics < 1 or periods < 1 or restarts < 1 or iterations < 0:
        raise ValueError("fourier_series_excitation needs fundamental_hz > 0, harmonics >= 1, periods >= 1, restarts >= 1, iterations >= 0")
    moved = space.joint_indices(joints)
    dof = space.dof
    centre = np.asarray(start, float)
    bounds = space.room(start)
    harmonic_numbers = np.arange(1, int(harmonics) + 1)
    omega = 2 * np.pi * fundamental_hz
    period_s = 1.0 / fundamental_hz
    rows = max(256, 32 * int(harmonics))  # the optimiser's grid over one period (analytic series, so this is generous)
    grid = np.arange(rows) * (period_s / rows)
    regressor_rows = np.arange(0, rows, max(1, rows // 64))  # the samples whose regressor enters the objective
    room = np.array([min(centre[j] - bounds.low[j], bounds.high[j] - centre[j]) for j in range(dof)])
    max_effort = getattr(space.limits, "max_effort", None)
    if max_effort is not None and not np.all(np.isfinite(max_effort)):
        max_effort = None

    def scaled(shape: np.ndarray) -> np.ndarray:
        """The coefficients, each moved joint's series scaled so its tightest of room, speed, acceleration is at ``margin``."""
        out = np.zeros((dof, 2 * len(harmonic_numbers)))
        x, dx, ddx = _series(shape, harmonic_numbers, omega, grid)
        for column, joint in enumerate(moved):
            ratios = (np.max(np.abs(x[:, column])) / max(room[joint], 1e-12), np.max(np.abs(dx[:, column])) / bounds.max_velocity[joint],
                      np.max(np.abs(ddx[:, column])) / bounds.max_acceleration[joint])
            out[joint] = shape[column] * margin / max(max(ratios), 1e-12)
        return out

    def states(full: np.ndarray, times: np.ndarray):
        x, dx, ddx = _series(full, harmonic_numbers, omega, times)
        return centre + x, dx, ddx

    def objective(shape: np.ndarray) -> tuple[float, np.ndarray]:
        full = scaled(shape)
        q, qd, qdd = states(full, grid)
        if max_effort is not None:  # a uniform gain, so the shape the optimiser chose is kept
            torque = np.array([space.torque(q[i], qd[i], qdd[i]) for i in regressor_rows])
            over = float(np.max(np.abs(torque) / max_effort))
            if over > margin:
                full = full * (margin / over)
                q, qd, qdd = states(full, grid)
        if space.checks_collision and any(space.in_collision(q[i]) for i in range(0, rows, 4)):
            return math.inf, full
        stack = np.vstack([_features(space, q[i], qd[i], qdd[i]) for i in regressor_rows])
        return _cost(stack, criterion), full

    rng = np.random.default_rng(seed)
    best_cost, best_full, best_shape = math.inf, None, None
    first_cost = math.inf
    for number in range(int(restarts)):
        shape = rng.standard_normal((len(moved), 2 * len(harmonic_numbers)))
        cost, full = objective(shape)
        if number == 0:
            first_cost = cost
        if cost < best_cost:
            best_cost, best_full, best_shape = cost, full, shape
    if best_full is None:
        raise ValueError("fourier_series_excitation: every random series collides or leaves the limits: lower margin or widen the room")
    step = 0.3
    for _ in range(int(iterations)):
        shape = best_shape + step * rng.standard_normal(best_shape.shape) * np.maximum(np.linalg.norm(best_shape, axis=1, keepdims=True), 1e-12) \
            / math.sqrt(best_shape.shape[1])
        cost, full = objective(shape)
        if cost < best_cost:
            best_cost, best_full, best_shape = cost, full, shape
        else:
            step = max(0.02, step * 0.995)
    t = k._samples(period_s * periods, space.rate_hz)
    w, dw, ddw = k._fade(t, t[-1], fade_s)
    x, dx, ddx = _series(best_full, harmonic_numbers, omega, t)
    x, dx, ddx = x * w[:, None], dx * w[:, None] + x * dw[:, None], ddx * w[:, None] + 2 * dx * dw[:, None] + x * ddw[:, None]
    # WHY a last uniform gain against the FULL limits (as signals.constant_speed_chirp): the fade's slope adds a little speed
    # and acceleration the optimiser's periodic series did not have; trimming the whole signal keeps its shape, not clipping.
    worst = max(float(np.max(np.abs(x[:, j])) / max(room[j], 1e-12)) for j in moved)
    worst = max(worst, float(np.max(np.abs(dx) / bounds.max_velocity)), float(np.max(np.abs(ddx) / bounds.max_acceleration)))
    gain = min(1.0, (1.0 - 1e-9) / worst)
    q, qd, qdd = centre + gain * x, gain * dx, gain * ddx
    real = np.vstack([_features(space, q[i], qd[i], qdd[i]) for i in range(0, len(t), max(1, len(t) // 400))])
    basis = "the model's regressor" if callable(getattr(space.dynamics, "regressor", None)) else "a generic basis from torque()"
    first = f" (the first random series: {math.exp(first_cost):.4g})" if criterion == "condition_number" else ""
    note = (f"{label}: {harmonics} harmonics of {fundamental_hz:g} Hz on {len(moved)} joints, {criterion.replace('_', ' ')} optimised over "
            f"{restarts} random series + {iterations} perturbations (seed {seed}); regressor {basis}; the run's regressor condition "
            f"number {condition_number(real):.4g}{first}")
    return Motion([k.Arrays(t, q, qd, qdd, label)], [note], entry={"criterion": criterion, "fundamental_hz": fundamental_hz})
