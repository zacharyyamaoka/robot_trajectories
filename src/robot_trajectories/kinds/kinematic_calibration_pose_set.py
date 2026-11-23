from __future__ import annotations

import math

import numpy as np

from ..registry import LIMITS, kind
from ..waypoint_trajectory_generator import WaypointTrajectoryGenerator
from ._lane_d import forward_or_none, line_clear, nearest_order, pose_set, usable

CRITERIA = ("observability_index", "noise_amplification", "min_singular_value", "condition_number")

# WHY the parameters are the joint zero offsets plus the tool's translation, and nothing else: the space injects forward kinematics
# only (no link-length parameter list), so the identification Jacobian is the one any forward model has: d(pose)/d(each joint's
# zero offset) by central differences, and d(point)/d(a tool offset in the flange frame) = the flange rotation. Zach's 2024-03
# notebook calibrated ~20 parameters (lengths plus backlash), not thousands; a robot whose kinematics exposes more passes a
# richer Jacobian through a richer kind.
OFFSET_SIGMA_RAD = 0.01  # WHY: the size of a zero error to expect (about 0.6 degrees): columns are scaled by it so rad and m compare
TOOL_SIGMA_M = 0.005  # ... and a 5 mm tool offset (Hollerbach & Wampler scale each parameter by its expected size)
ROTATION_WEIGHT_M = 0.1  # WHY: rad -> m for the orientation rows, as bam_calibration's rotation_weight (a 1 mrad error counts like 0.1 mm)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    skew = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0)))
    return 0.5 * skew if angle < 1e-9 else angle / (2.0 * math.sin(angle)) * skew


def identification_jacobian(space, q: np.ndarray, *, with_orientation: bool, rotation_weight_m: float = ROTATION_WEIGHT_M,
                            step_rad: float = 1e-6) -> np.ndarray | None:
    """The scaled Jacobian of the measured pose (position rows, then orientation rows weighted by ``rotation_weight_m``) with respect
    to the joint zero offsets (n columns) and the tool translation in the flange frame (3 columns), at ``q``; None where there is no pose."""
    pose = forward_or_none(space, q)
    if pose is None:
        return None
    n = len(q)
    rows = 6 if with_orientation else 3
    jacobian = np.zeros((rows, n + 3))
    for joint in range(n):
        plus, minus = np.array(q, float), np.array(q, float)
        plus[joint] += step_rad
        minus[joint] -= step_rad
        high, low = forward_or_none(space, plus), forward_or_none(space, minus)
        if high is None or low is None:
            return None
        jacobian[:3, joint] = (high[:3, 3] - low[:3, 3]) / (2.0 * step_rad) * OFFSET_SIGMA_RAD
        if with_orientation:
            jacobian[3:, joint] = rotation_weight_m * _rotation_vector(high[:3, :3] @ low[:3, :3].T) / (2.0 * step_rad) * OFFSET_SIGMA_RAD
    jacobian[:3, n:] = pose[:3, :3] * TOOL_SIGMA_M
    return jacobian


def index_of(criterion: str, stacked: np.ndarray, poses: int) -> float:
    """The observability index of a stacked identification Jacobian (rows x L, L observable parameters) from ``poses`` poses.

    Hollerbach & Wampler's taxonomy, as remembered (the papers are cited in the catalog row; not re-read for this port): with
    sigma_1 >= ... >= sigma_L the singular values, O1 = (sigma_1 ... sigma_L)^(1/L) / sqrt(m) (Borm & Menq), O2 = sigma_L / sigma_1,
    O3 = sigma_L, O4 = sigma_L^2 / sigma_1 (Nahvi & Hollerbach's noise amplification). Larger is better for every one."""
    sigma = np.linalg.svd(stacked, compute_uv=False)
    sigma = sigma[: stacked.shape[1]]
    if criterion == "observability_index":
        return float(np.exp(np.mean(np.log(np.maximum(sigma, 1e-300)))) / math.sqrt(poses))
    if criterion == "min_singular_value":
        return float(sigma[-1])
    if criterion == "condition_number":
        return float(sigma[-1] / sigma[0])
    return float(sigma[-1] ** 2 / sigma[0])  # noise_amplification


def _information_gain(jacobians: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """log det(I + J M^-1 J^T) for every candidate J (C, r, L): the D-optimal gain of adding it to an information matrix M."""
    solved = np.einsum("lk,crk->crl", inverse, jacobians)
    block = np.eye(jacobians.shape[1])[None] + np.einsum("crl,csl->crs", jacobians, solved)
    return np.linalg.slogdet(block)[1]


@kind(use_cases=("kinematic_calibration", "accuracy"), needs=(*LIMITS, "forward_kinematics"), unchecked=("collision_model",),
      catalog_id="kinematic_calibration_pose_set")
def kinematic_calibration_pose_set(space, *, poses=30, candidates=2000, criterion="observability_index", dwell_s=1.0,
                                   waypoint_generator=WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.5), seed=0,
                                   label="calibration poses", start=None):
    """Static poses chosen to make the calibration's identification Jacobian well conditioned, each held ``dwell_s`` for an
    external measurement (a pose set: stations joined by legs, labelled ``"<label>: station k"``).

    ``candidates`` configurations are drawn (seeded) inside the room around the start, kept if they have a tool pose, are free
    of collision and are reachable from the start along a clear joint-space line; ``poses`` of them are chosen by greedy
    D-optimal growth (each adds the most log-determinant to the information matrix) then exchange passes, and ordered by a
    nearest-neighbour tour. ``criterion`` names the index the exchange maximises ("observability_index" Borm & Menq's O1, or
    "noise_amplification", "min_singular_value", "condition_number"); the segment's entry records the index of the chosen set and
    the median of random sets of the same size, so the choice is held against chance.

    The parameters identified are the joint zero offsets and a tool translation (``# WHY`` in the source): the index is taken
    over the observable subspace of the candidate pool, so a parameter the forward model cannot see never zeroes it."""
    if criterion not in CRITERIA:
        raise ValueError(f"criterion is one of {CRITERIA}, got {criterion!r}")
    if poses < 1 or candidates < poses:
        raise ValueError(f"need 1 <= poses <= candidates, got poses={poses}, candidates={candidates}")
    rng = np.random.default_rng(seed)
    room = space.room(start)
    with_orientation = not space.position_only
    pool, jacobians, draws = [], [], 0
    while len(pool) < candidates and draws < 25 * candidates:
        draws += 1
        q = rng.uniform(room.low, room.high)
        if not usable(space, q, room) or not line_clear(space, start, q, room, samples=12):
            continue
        jacobian = identification_jacobian(space, q, with_orientation=with_orientation)
        if jacobian is None:
            continue
        pool.append(q)
        jacobians.append(jacobian)
    if len(pool) < poses:
        raise ValueError(f"{label}: only {len(pool)} usable configurations in {draws} draws inside the room: fewer than the {poses} asked")
    jacobians = np.array(jacobians)
    # the observable parameter subspace: right singular directions of the whole pool (a planar robot never sees the tool's height)
    _, sigma, vt = np.linalg.svd(jacobians.reshape(-1, jacobians.shape[-1]), full_matrices=False)
    observable = vt[sigma > 1e-8 * sigma[0]].T
    projected = jacobians @ observable
    count, rows, width = projected.shape
    ridge = 1e-9 * float(np.trace(projected[0].T @ projected[0]))

    def information(chosen) -> np.ndarray:
        return ridge * np.eye(width) + sum((projected[i].T @ projected[i] for i in chosen), np.zeros((width, width)))

    chosen: list[int] = []
    for _ in range(poses):  # greedy D-optimal growth
        gain = _information_gain(projected, np.linalg.inv(information(chosen)))
        gain[chosen] = -np.inf
        chosen.append(int(np.argmax(gain)))

    def score(subset) -> float:
        return index_of(criterion, np.vstack([projected[i] for i in subset]), len(subset))

    best = score(chosen)
    screen = rng.permutation(count)[: min(count, 300)]  # WHY a subset: the exchange scores each swap by an SVD
    for _ in range(2):  # exchange passes
        improved = False
        for slot in range(len(chosen)):
            for candidate in screen:
                if candidate in chosen:
                    continue
                trial = [*chosen[:slot], int(candidate), *chosen[slot + 1:]]
                value = score(trial)
                if value > best * (1.0 + 1e-9):
                    chosen, best, improved = trial, value, True
        if not improved:
            break
    baseline = float(np.median([score(list(rng.choice(count, size=poses, replace=False))) for _ in range(20)]))
    order = nearest_order(start, [pool[i] for i in chosen])
    stations = [pool[chosen[i]] for i in order]
    points = [space.forward(q)[:3, 3] for q in stations]
    notes = [f"{criterion} of the chosen {poses} poses {best:.4g}; the median of 20 random sets of {poses}: {baseline:.4g}",
             f"{width} observable parameters of {jacobians.shape[-1]} (joint zero offsets and a tool translation)"]
    return pose_set(space, label, start, stations, points=points, dwell_s=dwell_s, waypoint_generator=waypoint_generator,
                    room=room, via=start, notes=notes,
                    entry={"criterion": criterion, "index": best, "index_of_random_sets_median": baseline,
                           "observable_parameters": int(width), "candidates_used": int(count)})
