"""Lane C's shared helpers (docs/KINDS_TODO.md, "Lane C: point-to-point + coverage"): imported by lane C's kinds, registers nothing.

- the geometry of a circular arc (Pilz CIRC: by an interim point or by a centre) as a path ``s -> pose`` for ``cartesian.follow``;
- low-discrepancy and stratified sequences in numpy only (Sobol with Joe-Kuo direction numbers, Halton, Latin hypercube);
- the ZV / ZVD input shapers (Singer and Seering) and a shaped leg;
- a time parameterization that holds a tool speed (a trapezoid on the tool's own path length);
- the blended and spline joint paths of ``via_point_spline``;
- a faded sine (the tapered swing of the five-bar's coverage set).
"""

from __future__ import annotations

import math

import numpy as np
from robot_msgs.geometry.rotations import quat_from_matrix, quat_slerp, quat_to_matrix

from .. import kernels as k
from ..cartesian import Path, follow, line, rotation_angle
from ..time_parameterization import TimeParameterization

# ====================================================================== points and poses


def as_point(value, what: str) -> np.ndarray:
    """An xyz point (m) from an xyz or a 4x4 pose."""
    array = np.asarray(value, float)
    if array.shape == (4, 4):
        return array[:3, 3].copy()
    if array.shape == (3,):
        return array.copy()
    raise ValueError(f"{what} is an xyz point (3,) or a 4x4 pose, got shape {array.shape}")


def with_point(rotation: np.ndarray, point: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, point
    return pose


def tool_line(space, q: np.ndarray, target: np.ndarray, law: TimeParameterization, *, label: str, eef_step_m: float,
              jump_threshold: float, max_joint_step_rad: float, tolerance_m: float):
    """One straight tool line from where the tool is at ``q`` to the pose ``target``, rest to rest (``cartesian.follow``)."""
    here = space.forward(q)
    length = float(np.linalg.norm(target[:3, 3] - here[:3, 3]))
    turn = 0.0 if space.position_only else rotation_angle(here[:3, :3], target[:3, :3])
    # WHY 5 cm per radian (as cartesian_path): a pure turn still gets an inverse-kinematics sample every ~1 degree
    return follow(space, q, line(here, target), law, length_m=max(length, 0.05 * turn), label=label, eef_step_m=eef_step_m,
                  jump_threshold=jump_threshold, max_joint_step_rad=max_joint_step_rad, tolerance_m=tolerance_m)


def tool_speed_law(length_m: float, speed_m_s: float, ramp_time_s: float, *, min_move_s: float = 0.1) -> TimeParameterization:
    """A trapezoid on the tool's path length: ramp ``ramp_time_s`` up, cruise at ``speed_m_s``, ramp down (a triangle peaking at the
    speed when the line is too short to cruise).

    A leg of duration T with ramp fraction r covers v T (1 - r) at cruise speed v, so T = L / v + ramp and r = ramp / T; r is at most
    1/2 (a triangle, T = 2 L / v). The leg is still lengthened where a joint limit binds: the caller reads the speed it got."""
    if length_m <= 0 or speed_m_s <= 0:
        raise ValueError(f"a tool-speed law needs a positive length and speed, got {length_m} m at {speed_m_s} m/s")
    duration = length_m / speed_m_s + ramp_time_s
    fraction = ramp_time_s / duration
    if fraction > 0.5:
        duration, fraction = 2.0 * length_m / speed_m_s, 0.5
    return TimeParameterization.trapezoid(move_s=duration, ramp_fraction=max(fraction, 1e-3), min_move_s=min_move_s)


def peak_tool_speed(tool_target: np.ndarray, t: np.ndarray) -> float:
    """The peak speed (m/s) of the commanded tool point along a piece (finite differences of its ``tool_target`` channel)."""
    rows = np.isfinite(tool_target[:, 0])
    if np.count_nonzero(rows) < 2:
        return 0.0
    step = np.linalg.norm(np.diff(tool_target[rows], axis=0), axis=1) / np.diff(t[rows])
    return float(np.max(step))


# ====================================================================== arcs (Pilz CIRC)


def arc_geometry(start: np.ndarray, goal: np.ndarray, *, interim: np.ndarray | None, centre: np.ndarray | None,
                 long_way: bool = False, radius_tolerance_m: float = 1e-4):
    """(centre, normal, radius, sweep rad) of the circular arc from ``start`` to ``goal``.

    By an ``interim`` point: the circle through the three points, the arc running start -> interim -> goal (so ``sweep`` is signed
    about ``normal`` and may exceed pi). By a ``centre``: the circle about it, the short way round (``long_way`` the other way);
    start and goal must both lie on it. Refused, never repaired: three collinear points, a centre off the circle, a diameter
    (whose plane the centre cannot name)."""
    if (interim is None) == (centre is None):
        raise ValueError("cartesian_arc: give exactly one of interim= (a point the arc passes through) or centre=")
    if np.linalg.norm(goal - start) < 1e-9:
        raise ValueError("cartesian_arc: the goal is where the tool already is")
    if interim is not None:
        edge_b, edge_c = interim - start, goal - start
        normal = np.cross(edge_b, edge_c)
        area = float(np.dot(normal, normal))
        if area < 1e-12 * float(np.dot(edge_b, edge_b)) * float(np.dot(edge_c, edge_c)) or area < 1e-18:
            raise ValueError("cartesian_arc: the start, interim and goal points lie on one line: no circle through them")
        centre = start + np.cross(np.dot(edge_b, edge_b) * edge_c - np.dot(edge_c, edge_c) * edge_b, normal) / (2.0 * area)
        normal = normal / math.sqrt(area)
        radius = float(np.linalg.norm(start - centre))
        u = (start - centre) / radius
        v = np.cross(normal, u)
        sweep = math.atan2(float(np.dot(goal - centre, v)), float(np.dot(goal - centre, u))) % (2.0 * math.pi)  # start -> interim -> goal turns +
        return centre, normal, radius, sweep
    spoke_a, spoke_b = start - centre, goal - centre
    radius = float(np.linalg.norm(spoke_a))
    if abs(float(np.linalg.norm(spoke_b)) - radius) > radius_tolerance_m or radius < 1e-9:
        raise ValueError(f"cartesian_arc: the start is {1000 * radius:.3f} mm and the goal {1000 * np.linalg.norm(spoke_b):.3f} mm "
                         f"from the centre; both must lie on the circle about it (within {1000 * radius_tolerance_m:.1f} mm)")
    normal = np.cross(spoke_a, spoke_b)
    size = float(np.linalg.norm(normal))
    if size < 1e-9 * radius * radius:
        raise ValueError("cartesian_arc: the start, goal and centre lie on one line (a half circle): the centre does not name its "
                         "plane; give an interim point instead")
    normal = normal / size
    short = math.atan2(size, float(np.dot(spoke_a, spoke_b)))  # (0, pi): the short way round, positive about the normal
    return centre, normal, radius, (short - 2.0 * math.pi) if long_way else short


def arc_path(start_pose: np.ndarray, goal_pose: np.ndarray, centre: np.ndarray, normal: np.ndarray, radius: float, sweep: float,
             orientation: str) -> Path:
    """The tool path on that arc, progress s in [0, 1]: position on the circle (the geometry), orientation held or slerped."""
    u = (start_pose[:3, 3] - centre) / radius
    v = np.cross(normal, u)
    qa, qb = quat_from_matrix(start_pose[:3, :3]), quat_from_matrix(goal_pose[:3, :3])

    def at(s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, float)
        angle = sweep * s
        poses = np.tile(np.eye(4), (len(s), 1, 1))
        poses[:, :3, 3] = centre + radius * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v)
        if orientation == "follow":
            poses[:, :3, :3] = quat_to_matrix(quat_slerp(qa, qb, s))
        else:
            poses[:, :3, :3] = start_pose[:3, :3]
        return poses

    return at


# ====================================================================== low-discrepancy and stratified sequences (numpy only)

SOBOL_BITS = 30
# Joe and Kuo (2008), new-joe-kuo-6.21201: dimension -> (degree s, polynomial coefficients a, initial direction numbers m)
SOBOL_NUMBERS = {2: (1, 0, (1,)), 3: (2, 1, (1, 3)), 4: (3, 1, (1, 3, 1)), 5: (3, 2, (1, 1, 1)), 6: (4, 1, (1, 1, 3, 3)),
                 7: (4, 4, (1, 3, 5, 13)), 8: (5, 2, (1, 1, 5, 5, 17)), 9: (5, 4, (1, 1, 5, 5, 5)), 10: (5, 7, (1, 1, 7, 11, 19))}
HALTON_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)
SEQUENCES = ("sobol", "halton", "latin_hypercube")


def _sobol_directions(dimension: int) -> list[int]:
    """The direction integers v_1..v_BITS (scaled by 2^BITS) of one Sobol dimension (1-based)."""
    if dimension == 1:
        return [1 << (SOBOL_BITS - i) for i in range(1, SOBOL_BITS + 1)]
    degree, coefficients, initial = SOBOL_NUMBERS[dimension]
    v = [0] + [m << (SOBOL_BITS - i) for i, m in enumerate(initial, start=1)]
    for i in range(degree + 1, SOBOL_BITS + 1):
        value = v[i - degree] ^ (v[i - degree] >> degree)
        for j in range(1, degree):
            if (coefficients >> (degree - 1 - j)) & 1:
                value ^= v[i - j]
        v.append(value)
    return v[1:]


def sobol(count: int, dimensions: int, seed: int | None = None) -> np.ndarray:
    """``count`` Sobol points in [0, 1)^dimensions (Joe-Kuo numbers, Gray-code order; the first is the origin). With ``seed`` a
    random digital shift (an XOR with seeded random bits per dimension) keeps every dyadic stratification and moves the points off
    the corner. Sobol wants ``count`` a power of two (a balanced set); other counts are allowed and unbalanced."""
    if dimensions > max(SOBOL_NUMBERS):
        raise ValueError(f"sobol covers up to {max(SOBOL_NUMBERS)} joints here (Joe-Kuo numbers kept in the library); "
                         f"this robot has {dimensions}: use method='halton' or 'latin_hypercube'")
    directions = [_sobol_directions(d) for d in range(1, dimensions + 1)]
    integers = np.zeros((count, dimensions), np.int64)
    state = [0] * dimensions
    for n in range(1, count):
        bit = (~(n - 1) & n).bit_length() - 1  # the lowest zero bit of n - 1 (Gray-code order)
        for d in range(dimensions):
            state[d] ^= directions[d][bit]
        integers[n] = state
    if seed is not None:
        shift = np.random.default_rng(seed).integers(0, 1 << SOBOL_BITS, size=dimensions, dtype=np.int64)
        integers = integers ^ shift
    return integers / float(1 << SOBOL_BITS)


def halton(count: int, dimensions: int, seed: int | None = None) -> np.ndarray:
    """``count`` Halton points (radical inverse in the first primes, from index 1); ``seed`` a Cranley-Patterson random shift."""
    if dimensions > len(HALTON_PRIMES):
        raise ValueError(f"halton covers up to {len(HALTON_PRIMES)} joints here, this robot has {dimensions}: use 'latin_hypercube'")
    points = np.zeros((count, dimensions))
    for d, base in enumerate(HALTON_PRIMES[:dimensions]):
        for n in range(count):
            index, fraction, value = n + 1, 1.0 / base, 0.0
            while index:
                index, digit = divmod(index, base)
                value += digit * fraction
                fraction /= base
            points[n, d] = value
    if seed is not None:
        points = (points + np.random.default_rng(seed).random(dimensions)) % 1.0
    return points


def latin_hypercube(count: int, dimensions: int, seed: int = 0) -> np.ndarray:
    """``count`` Latin-hypercube points (McKay et al.): each dimension one point per 1/count stratum, strata paired at random."""
    rng = np.random.default_rng(seed)
    cells = np.stack([rng.permutation(count) for _ in range(dimensions)], axis=1)
    return (cells + rng.random((count, dimensions))) / count


def unit_points(method: str, count: int, dimensions: int, seed: int) -> np.ndarray:
    if method == "sobol":
        return sobol(count, dimensions, seed)
    if method == "halton":
        return halton(count, dimensions, seed)
    if method == "latin_hypercube":
        return latin_hypercube(count, dimensions, seed)
    raise ValueError(f"method is one of {SEQUENCES}, got {method!r}")


# ====================================================================== input shapers (Singer and Seering, 1990)

SHAPERS = ("ZV", "ZVD")


def shaper_impulses(shaper: str, mode_hz: float, damping_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """(times, amplitudes) of a ZV or ZVD shaper for a mode of damped frequency ``mode_hz`` and damping ratio ``damping_ratio``:
    impulses spaced half a damped period, amplitudes summing to 1 (zero residual vibration at the modelled mode; ZVD also zero
    first sensitivity to its frequency)."""
    if shaper not in SHAPERS:
        raise ValueError(f"shaper is one of {SHAPERS}, got {shaper!r}")
    if mode_hz <= 0 or not 0.0 <= damping_ratio < 1.0:
        raise ValueError(f"a mode needs mode_hz > 0 and 0 <= damping_ratio < 1, got {mode_hz} Hz, {damping_ratio}")
    ratio = math.exp(-damping_ratio * math.pi / math.sqrt(1.0 - damping_ratio**2))
    half = 0.5 / mode_hz
    if shaper == "ZV":
        return np.array([0.0, half]), np.array([1.0, ratio]) / (1.0 + ratio)
    return np.array([0.0, half, 2.0 * half]), np.array([1.0, 2.0 * ratio, ratio**2]) / (1.0 + ratio) ** 2


def shaped_leg(q0: np.ndarray, q1: np.ndarray, law: TimeParameterization, limits, rate_hz: float,
               impulses: list[tuple[np.ndarray, np.ndarray]], *, label: str) -> k.Arrays:
    """The rest-to-rest leg q0 -> q1 under a built-in ``law``, each joint convolved with its own impulse sequence: the shaped
    motion is the sum of A_i x (the leg delayed by t_i). A convex combination of one leg's shifted copies, so positions stay between
    q0 and q1 and no speed or acceleration peak exceeds the leg's."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    timing = law.leg_timing(q0, q1, limits)
    duration = timing.duration_s
    tail = max(float(times[-1]) for times, _ in impulses)
    steps = max(2, int(math.ceil((duration + tail) * rate_hz - 1e-9)))
    t = np.arange(steps + 1) / rate_hz
    s = np.zeros((len(t), len(q0)))
    ds = np.zeros_like(s)
    dds = np.zeros_like(s)
    for joint, (times, amplitudes) in enumerate(impulses):
        for delay, amplitude in zip(times, amplitudes, strict=True):
            tau = (t - delay) / duration
            inside = (tau > 0.0) & (tau < 1.0)
            progress, speed, acceleration = law.profile(np.clip(tau, 0.0, 1.0))
            s[:, joint] += amplitude * progress
            ds[:, joint] += amplitude * np.where(inside, speed, 0.0) / duration
            dds[:, joint] += amplitude * np.where(inside, acceleration, 0.0) / duration**2
    delta = q1 - q0
    q = q0 + delta * s
    q[-1] = q1  # WHY: the last sample is the target exactly, not the sum of amplitudes' rounding
    return k.Arrays(t, q, delta * ds, delta * dds, label)


# ====================================================================== via points: blended corners and a spline


def uniform_arc(path: np.ndarray, spacing: float = 0.001, *, minimum: int = 1000, maximum: int = 12000) -> np.ndarray:
    """``path`` (a fine polyline) resampled at equal arc-length steps (about ``spacing`` apart): the grid a time parameterization differentiates
    on, so its slope and curvature are clean (a uniform grid, no jump in spacing where a side meets a blend)."""
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    count = int(np.clip(math.ceil(chord[-1] / spacing) + 1, minimum, maximum))
    grid = np.linspace(0.0, chord[-1], count)
    out = np.stack([np.interp(grid, chord, path[:, joint]) for joint in range(path.shape[1])], axis=1)
    out[0], out[-1] = path[0], path[-1]
    return out


def blend_polyline(points: np.ndarray, radii: np.ndarray, *, samples: int = 600, max_step: float = 0.0005) -> np.ndarray:
    """A joint-space path through ``points`` with each inner corner i cut within ``radii[i]`` (the distance, in joint space, along
    each side from the corner where the blend starts and ends): straight sides, and at each corner a quintic Bezier whose end
    curvature is zero, so the path's tangent AND curvature are continuous (a smooth acceleration, a finite jerk). The ends are the
    first and last points exactly. Pilz's blend: the path leaves the corner's neighbourhood by a smooth curve instead of stopping."""
    path = [points[0]]
    here = points[0]
    for index in range(1, len(points) - 1):
        corner, before, after = points[index], points[index - 1], points[index + 1]
        radius = float(radii[index])
        enter = corner + radius * (before - corner) / np.linalg.norm(before - corner)
        leave = corner + radius * (after - corner) / np.linalg.norm(after - corner)
        _append_line(path, here, enter, max_step)
        # WHY P2 = P3 = the corner and P1, P4 halfway along each side: P0, P1, P2 are collinear (zero curvature at the start of the
        # blend), so the join with the straight side is C2 as a curve
        controls = [enter, enter + 0.5 * (corner - enter), corner, corner, leave + 0.5 * (corner - leave), leave]
        u = np.linspace(0.0, 1.0, samples)[1:, None]
        bernstein = [math.comb(5, i) * u**i * (1 - u) ** (5 - i) for i in range(6)]
        path.extend(sum(weight * control for weight, control in zip(bernstein, controls, strict=True)))
        here = leave
    _append_line(path, here, points[-1], max_step)
    return np.array(path)


def _append_line(path: list, a: np.ndarray, b: np.ndarray, max_step: float) -> None:
    count = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / max_step)))
    for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
        path.append(a + fraction * (b - a))


def natural_spline(points: np.ndarray, *, per_chord: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """A natural cubic spline through ``points`` (chord-length parameter): (dense path, its chord-length parameter). Passes
    through every point exactly; C2 at them."""
    points = np.asarray(points, float)
    chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
    h = chords
    n = len(points)
    if n == 2:
        param = np.linspace(0.0, h[0], per_chord)
        return points[0] + (points[1] - points[0]) * (param / h[0])[:, None], param
    matrix = np.zeros((n, n))
    rhs = np.zeros_like(points)
    matrix[0, 0] = matrix[-1, -1] = 1.0
    for i in range(1, n - 1):
        matrix[i, i - 1], matrix[i, i], matrix[i, i + 1] = h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 6.0 * ((points[i + 1] - points[i]) / h[i] - (points[i] - points[i - 1]) / h[i - 1])
    second = np.linalg.solve(matrix, rhs)
    dense, param = [points[0]], [0.0]
    offset = 0.0
    for i in range(n - 1):
        local = np.linspace(0.0, h[i], per_chord)[1:]
        a, b = (h[i] - local)[:, None], local[:, None]
        value = (second[i] * a**3 + second[i + 1] * b**3) / (6 * h[i]) + (points[i] / h[i] - second[i] * h[i] / 6) * a \
            + (points[i + 1] / h[i] - second[i + 1] * h[i] / 6) * b
        dense.extend(value)
        param.extend(offset + local)
        offset += h[i]
    return np.array(dense), np.array(param)


# ====================================================================== a faded sine


def faded_sine(centre: np.ndarray, joint: int, amplitude: float, hz: float, ramp_cycles: float, full_cycles: float, rate_hz: float,
               *, label: str) -> k.Arrays:
    """centre + A w(t) sin(2 pi f t) on one joint (others held), w a raised-cosine ramp ``ramp_cycles`` long in and out, so the swing
    starts and ends at rest at its centre (the five-bar coverage set's swing, bam_fivebar coverage_set.swing_motors)."""
    period = 1.0 / hz
    total = (2.0 * ramp_cycles + full_cycles) * period
    steps = max(2, int(round(total * rate_hz)))
    t = np.arange(steps + 1) / rate_hz
    total = t[-1]
    ramp = ramp_cycles * period
    w, dw, ddw = k._fade(t, total, ramp)  # the kernels' raised cosine in and out, and its two derivatives
    omega = 2.0 * math.pi * hz
    x, dx, ddx = np.sin(omega * t), omega * np.cos(omega * t), -(omega**2) * np.sin(omega * t)
    q = np.tile(np.asarray(centre, float), (len(t), 1))
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    q[:, joint] += amplitude * x * w
    qd[:, joint] = amplitude * (dx * w + x * dw)
    qdd[:, joint] = amplitude * (ddx * w + 2.0 * dx * dw + x * ddw)
    return k.Arrays(t, q, qd, qdd, label)
