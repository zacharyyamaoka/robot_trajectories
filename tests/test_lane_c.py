"""Lane C's kinds (point-to-point + coverage), each held to the kind contract on the robots it applies to, and to what it is for.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_lane_c.py

Every Cartesian kind is also held to the tool points it commanded by forward kinematics of the output (within 1 mm) and refuses an
unreachable point; each kind has one claim about what it is FOR that can fail (a trapezoid's shape, a shaped move's residual
vibration, a pose set's stratification, a raster's coverage, ...).
"""

from __future__ import annotations

import math

import numpy as np

import _runner
import scenario
from fixtures import lane_c, robots
from kind_contract import check_kind
from robot_trajectories import (JointSpace, bands, MissingRung, TimeLaw, Unreachable, approach_retreat, cartesian_a_to_b, cartesian_arc,
                                cartesian_raster, chain, coverage_greedy_swings, input_shaped_move, kinds, labels, lissajous_multi_joint,
                                notes, polynomial_move, segment_info, space_filling_pose_set, trapezoidal_velocity, via_point_spline)
from robot_trajectories.kinds import _lane_c
from robot_trajectories.segments import TOOL_TARGET

MM = 0.001
FIVEBAR_START = np.radians([128.0, -126.0])


def pendulum() -> JointSpace:
    return scenario.pendulum_space()


def fivebar_joint() -> JointSpace:  # a reach map, no kinematics: the joint-space kinds
    return scenario.fivebar_space()


def fivebar() -> JointSpace:  # kinematics, no reach map (test_cartesian's)
    return JointSpace.from_limits(robots.FIVEBAR_LIMITS, home=np.radians([100.0, -76.0])).with_kinematics(robots.FiveBarKinematics())


def fivebar_reach() -> JointSpace:  # kinematics and the reach map as a cache
    return scenario.fivebar_space().with_kinematics(robots.FiveBarKinematics())


def ur5e() -> JointSpace:
    return scenario.ur5e_space().with_kinematics(robots.UR5eKinematics())


def with_dynamics(space: JointSpace) -> JointSpace:
    if space.dof == 1:
        return space.with_dynamics(robots.PendulumDynamics())
    return space.with_dynamics(lane_c.diagonal_dynamics(space.dof))  # STAND-IN dynamics


def tracks(space: JointSpace, trajectory) -> float:
    """The worst distance (m) between forward kinematics of the output and the tool point it commanded, where it commanded one."""
    target = trajectory.channels[TOOL_TARGET]
    rows = np.flatnonzero(np.isfinite(target[:, 0]))
    assert len(rows) > 0, "no tool point was commanded"
    return float(np.max(np.linalg.norm(space.tool_points(trajectory)[rows] - target[rows], axis=1)))


def reachable_fivebar(trajectory) -> None:
    off = [q for q in trajectory.positions[::5] if (passive := robots.fivebar_passive(*q)) is None or passive[2] < 0.2]
    assert not off, f"the five-bar leaves its reachable set (fold margin < 0.2) at {len(off)} samples"


def refused(call, words: str) -> None:
    try:
        call()
    except (ValueError, MissingRung) as error:  # Unreachable and MissingRung are ValueErrors
        assert words in str(error), f"refused, but not saying {words!r}: {error}"
    else:
        raise AssertionError(f"not refused (expected {words!r})")


# ---------------------------------------------------------------------- trapezoidal_velocity


def test_trapezoidal_velocity_is_a_scaled_synchronised_trapezoid_on_every_robot():
    for space in (pendulum(), fivebar_joint(), scenario.ur5e_space()):
        for fractions in ((1.0, 1.0), (0.5, 0.25)):
            move = trapezoidal_velocity(space, velocity_fraction=fractions[0], acceleration_fraction=fractions[1])
            check_kind(move, space)
            limits = space.limits
            ratio_v = np.max(np.abs(move.velocities) / (fractions[0] * limits.max_velocity))
            ratio_a = np.max(np.abs(move.accelerations) / (fractions[1] * limits.max_acceleration))
            assert ratio_v <= 1 + 1e-6 and ratio_a <= 1 + 1e-6, (ratio_v, ratio_a)
            assert max(ratio_v, ratio_a) >= 0.98, "not as fast as the scaled limits allow"  # time-optimal for the profile
            travel = move.positions - move.positions[0]
            delta = move.positions[-1] - move.positions[0]
            progress = travel @ delta / (delta @ delta)
            assert np.max(np.abs(travel - progress[:, None] * delta)) < 1e-9, "the joints do not arrive together (a straight joint line)"
            # the trapezoid: speed constant through the middle half, zero at the ends
            speed = np.linalg.norm(move.velocities, axis=1)
            middle = speed[int(0.3 * len(speed)):int(0.7 * len(speed))]
            assert np.ptp(middle) <= 1e-6 * speed.max(), "no cruise"
            assert speed[0] == 0 and speed[-1] == 0
    space = scenario.ur5e_space()
    slow = trapezoidal_velocity(space, velocity_fraction=0.2)
    fast = trapezoidal_velocity(space, velocity_fraction=1.0)
    assert slow.t[-1] > fast.t[-1], "a lower velocity fraction is not slower"
    for bad in (0.0, 1.5):
        refused(lambda bad=bad: trapezoidal_velocity(space, velocity_fraction=bad), "velocity_fraction")
    refused(lambda: trapezoidal_velocity(space, np.zeros(6)), "outside its limits")


# ---------------------------------------------------------------------- polynomial_move


def test_polynomial_move_cubic_and_quintic_hit_their_states_and_hand_a_moving_end_on():
    for space in (pendulum(), fivebar_joint(), scenario.ur5e_space()):
        for order in (3, 5):
            move = polynomial_move(space, order=order, duration_s=1.5)
            check_kind(move, space)
            assert abs(move.t[-1] - 1.5) <= 1e-9 or move.t[-1] > 1.5
            assert np.allclose(move.velocities[[0, -1]], 0, atol=1e-9)
        # quintic: the minimum-jerk leg when rest to rest (the same positions as TimeLaw.minimum_jerk's)
        leg = TimeLaw.minimum_jerk(move_s=1.5).leg(space.home, space.home + 0.5 * (space.room(space.home).high - space.home),
                                                   space.limits, space.rate_hz)
        assert np.allclose(polynomial_move(space, order=5, duration_s=1.5).positions, leg.q, atol=1e-12)
    space = scenario.ur5e_space()
    cruise = 0.3 * np.ones(6)
    first = polynomial_move(space, order=5, duration_s=1.0, end_velocity_rad_s=cruise)
    second = polynomial_move(space, target=space.home, order=3, duration_s=1.0, start=first)
    assert np.allclose(first.velocities[-1], cruise) and np.allclose(second.velocities[0], cruise), "the moving state was not handed on"
    both = chain(first, second)
    assert np.allclose(both.velocities[len(first.t) - 1], cruise)
    check_kind(both, space)
    # a cubic matches position and velocity at both ends, a quintic also the acceleration
    quintic = polynomial_move(space, order=5, duration_s=2.0, end_acceleration_rad_s2=np.ones(6))
    assert np.allclose(quintic.accelerations[-1], 1.0, atol=1e-6)
    short = polynomial_move(space, order=5, duration_s=0.05)
    assert short.t[-1] > 0.05 and any("limit binds" in note for note in notes(short)), "an infeasible duration is lengthened, and says so"
    refused(lambda: polynomial_move(space, order=4), "order")


# ---------------------------------------------------------------------- via_point_spline


def via_points(space: JointSpace) -> list[np.ndarray]:
    room, home = space.room(space.home), space.home
    up, down = room.high - home, room.low - home
    return [home + 0.5 * up * (np.arange(space.dof) % 2), home + 0.4 * up - 0.3 * down * (np.arange(space.dof) < 1), home + 0.45 * down, home]


def test_via_point_spline_passes_through_without_stopping():
    for space in (fivebar_joint(), scenario.ur5e_space()):
        vias = via_points(space)
        for interpolation in ("blend", "spline"):
            motion = via_point_spline(space, vias, interpolation=interpolation, blend_radius_rad=0.05)
            check_kind(motion, space)
            speed = np.linalg.norm(motion.velocities, axis=1)
            for via in vias[:-1]:
                nearest = int(np.argmin(np.linalg.norm(motion.positions - via, axis=1)))
                miss = np.linalg.norm(motion.positions[nearest] - via)
                assert speed[nearest] > 0.15 * speed.max(), f"{interpolation}: it slows to {speed[nearest]:.3f} at a via point"
                limit = 0.06 if interpolation == "blend" else 1e-3  # a blend cuts the corner within its radius; a spline passes through
                assert miss <= limit * (1.5 if interpolation == "blend" else 1.0), f"{interpolation}: {miss:.4f} rad from a via point"
            assert np.allclose(motion.positions[-1], vias[-1])
    # stop_at: rest, and the dwell, exactly there
    space = scenario.ur5e_space()
    vias = via_points(space)
    stopped = via_point_spline(space, vias, stop_at=(1,), dwell_s=0.2)
    check_kind(stopped, space)
    at_rest = np.flatnonzero(np.linalg.norm(stopped.velocities, axis=1) == 0)
    assert any(np.allclose(stopped.positions[row], vias[1]) for row in at_rest), "stop_at did not stop at its via point"
    # a stated jerk limit is held (the sampled jerk), a missing one noted
    limits = robots.UR5E_DEMO_BOX.replace(max_jerk=np.full(6, 40.0))
    jerky = JointSpace.from_limits(limits)
    smooth = via_point_spline(jerky, vias, blend_radius_rad=0.3)
    check_kind(smooth, jerky)
    jerk = np.max(np.abs(np.diff(smooth.accelerations, axis=0)) * jerky.rate_hz)
    assert jerk <= 40.0 * 1.0, f"jerk {jerk:.1f} over its limit of 40"
    assert any("jerk" in note for note in notes(via_point_spline(space, vias))), "no note that the jerk is not limited"
    refused(lambda: via_point_spline(space, [np.zeros(6)]), "outside its limits")
    refused(lambda: via_point_spline(space, vias, interpolation="line"), "interpolation")
    refused(lambda: via_point_spline(jerky, vias, time_law=TimeLaw.trapezoid(peak_speed_fraction=0.5)), "unlimited jerk")
    refused(lambda: via_point_spline(space, vias, blend_radius_m=0.05), "kinematics")
    metric = via_point_spline(ur5e(), vias, blend_radius_m=0.05)  # Pilz's way: the tool metres from the corner
    check_kind(metric, ur5e())


# ---------------------------------------------------------------------- cartesian_arc


def on_circle(space: JointSpace, trajectory, centre: np.ndarray, radius: float) -> float:
    """The independent oracle: how far (m) the output's tool points are from the circle (a distance from the centre = the radius)."""
    points = space.tool_points(trajectory)
    return float(np.max(np.abs(np.linalg.norm(points - centre, axis=1) - radius)))


def test_cartesian_arc_follows_a_circle_by_interim_point_and_by_centre():
    for space, start in ((fivebar(), FIVEBAR_START), (ur5e(), None)):
        start_q = space.home if start is None else start
        p = space.forward(start_q)[:3, 3]
        goal, interim = p + (0.04, 0.0, 0.0), p + (0.02, 0.02, 0.0)  # the circle through them: centre p + (0.02, 0, 0), radius 0.02
        centre = p + (0.02, 0.0, 0.0)
        by_interim = cartesian_arc(space, goal, interim=interim, start=start)
        check_kind(by_interim, space)
        assert tracks(space, by_interim) <= MM
        assert on_circle(space, by_interim, centre, 0.02) <= MM, "off the circle through the three points"
        points = space.tool_points(by_interim)
        assert np.min(np.linalg.norm(points - interim, axis=1)) <= MM, "does not pass through its interim point"
        assert np.all(points[:, 1] >= p[1] - MM), "went round the wrong side of the circle"
        quarter = p + (0.02, 0.02, 0.0)
        by_centre = cartesian_arc(space, quarter, centre=centre, start=start)
        check_kind(by_centre, space)
        assert tracks(space, by_centre) <= MM and on_circle(space, by_centre, centre, 0.02) <= MM
        assert np.max(space.tool_points(by_centre)[:, 1]) <= p[1] + 0.02 + MM and np.all(space.tool_points(by_centre)[:, 1] >= p[1] - MM), \
            "the short way round a quarter circle left its quadrant"
        # the long way round is the other three quarters, through the far side of the circle
        long_way = cartesian_arc(space, quarter, centre=centre, long_way=True, start=start)
        check_kind(long_way, space)
        assert on_circle(space, long_way, centre, 0.02) <= MM
        assert np.min(space.tool_points(long_way)[:, 1]) < p[1] - 0.015, "the long way did not go round the far side"
        # the interim point names the side: below the line the arc goes below it
        below = cartesian_arc(space, goal, interim=p + (0.02, -0.02, 0.0), start=start)
        assert np.all(space.tool_points(below)[:, 1] <= p[1] + MM), "the other side's arc went the wrong way"
        # a half circle by centre is refused (the centre does not name the plane)
        refused(lambda: cartesian_arc(space, goal, centre=centre, start=start), "one line")
        # refused: collinear, off the circle, unreachable, a centre without a goal
        refused(lambda: cartesian_arc(space, p + (0.04, 0.0, 0.0), interim=p + (0.02, 0.0, 0.0), start=start), "one line")
        refused(lambda: cartesian_arc(space, goal, centre=p + (0.03, 0.01, 0.0), start=start), "circle about it")
        refused(lambda: cartesian_arc(space, p + (3.0, 0.0, 0.0), interim=p + (1.5, 0.5, 0.0), start=start), "unreachable")
        refused(lambda: cartesian_arc(space, goal, start=start, interim=interim, centre=centre), "exactly one")
        assert isinstance(cartesian_arc(space, start=start), type(by_interim)), "callable as f(space) alone"
    # the UR5e follows a pose: the orientation slerps to the goal's own
    space = ur5e()
    home_pose = space.forward(space.home)
    goal = home_pose.copy()
    goal[:3, 3] += (0.04, 0.0, 0.0)
    twist = np.array([[math.cos(0.2), -math.sin(0.2), 0], [math.sin(0.2), math.cos(0.2), 0], [0, 0, 1.0]])
    goal[:3, :3] = twist @ goal[:3, :3]
    turned = cartesian_arc(space, goal, interim=home_pose[:3, 3] + (0.02, 0.02, 0.0), orientation="follow")
    check_kind(turned, space)
    last = space.forward(turned.positions[-1])
    assert np.allclose(last[:3, :3], goal[:3, :3], atol=2e-3), "did not end in the goal's orientation"


# ---------------------------------------------------------------------- approach_retreat


def test_approach_retreat_goes_out_and_back_along_the_axis():
    for space, start in ((fivebar(), FIVEBAR_START), (ur5e(), None)):
        start_q = space.home if start is None else start
        pose = space.forward(start_q)
        axis = pose[:3, 0] if space.position_only else pose[:3, 2]
        both = approach_retreat(space, distance_m=0.03, start=start)
        check_kind(both, space)
        assert tracks(space, both) <= MM
        points = space.tool_points(both)
        along = (points - pose[:3, 3]) @ axis
        across = np.linalg.norm((points - pose[:3, 3]) - along[:, None] * axis, axis=1)
        assert across.max() <= MM, "left the axis"
        assert abs(along.max() - 0.03) <= MM and np.allclose(points[-1], pose[:3, 3], atol=MM), "not out 3 cm and back"
        assert [band.label for band in bands(both)] == ["approach and retreat: approach", "approach and retreat: dwell",
                                                          "approach and retreat: retreat"], [band.label for band in bands(both)]
        out_only = approach_retreat(space, distance_m=0.03, approach=False, dwell_s=0.0, start=start)
        check_kind(out_only, space)
        final = space.tool_points(out_only)[-1]
        assert np.allclose(final, pose[:3, 3] - 0.03 * axis, atol=MM), "out only did not end 3 cm back along the axis"
        explicit = approach_retreat(space, distance_m=0.02, axis=(0.0, 0.0, 1.0) if not space.position_only else (0.0, -1.0, 0.0),
                                    retreat=False, start=start)
        check_kind(explicit, space)
        refused(lambda: approach_retreat(space, distance_m=3.0, start=start), "unreachable")
        refused(lambda: approach_retreat(space, approach=False, retreat=False, start=start), "approach or retreat")
    if True:
        space = ur5e()
        orientation = [space.forward(q)[:3, :3] for q in approach_retreat(space, distance_m=0.03).positions[::50]]
        assert max(_lane_c_angle(orientation[0], other) for other in orientation) < math.radians(0.1), "the orientation turned"


def _lane_c_angle(a: np.ndarray, b: np.ndarray) -> float:
    from robot_trajectories.cartesian import rotation_angle

    return rotation_angle(a, b)


# ---------------------------------------------------------------------- cartesian_a_to_b


def test_cartesian_a_to_b_draws_reachable_poses_and_joins_them_in_joint_space():
    for space, start in ((fivebar_reach(), FIVEBAR_START), (ur5e(), None)):
        a = cartesian_a_to_b(space, seed=3, start=start)
        check_kind(a, space)
        assert tracks(space, a) <= MM
        if space.reach is not None:
            reachable_fivebar(a)
            assert all(space.reach.nearest_is_reachable(q) for q in a.positions[::5]), "a leg left the reach map"
        assert np.array_equal(a.positions, cartesian_a_to_b(space, seed=3, start=start).positions), "the same seed is the same motion"
        b = cartesian_a_to_b(space, seed=4, start=start)
        assert not np.array_equal(a.positions, b.positions), "another seed drew the same pose"
        farther = space.tool_points(a)[-1] - space.forward(space.home if start is None else start)[:3, 3]
        assert np.linalg.norm(farther) >= 0.02 - MM
        many = cartesian_a_to_b(space, seed=5, n_intermediate_points=2, start=start)
        check_kind(many, space)
        assert len([band for band in bands(many) if ": leg" in band.label]) == 3
        # a given goal: reached to 1 mm; a goal the arm cannot reach is refused
        p = space.forward(space.home if start is None else start)[:3, 3]
        given = cartesian_a_to_b(space, p + (0.03, -0.02 if space.position_only else 0.02, 0.0), start=start)
        check_kind(given, space)
        assert tracks(space, given) <= MM
        refused(lambda: cartesian_a_to_b(space, p + (3.0, 0.0, 0.0), start=start), "unreachable")
    space = ur5e()
    p = space.forward(space.home)[:3, 3]
    via = cartesian_a_to_b(space, p + (0.1, 0.0, 0.0), via=[p + (0.05, 0.05, 0.0)])
    check_kind(via, space)
    assert tracks(space, via) <= MM


# ---------------------------------------------------------------------- input_shaped_move


def test_input_shaped_move_leaves_no_vibration_in_the_mode_it_is_tuned_to():
    for space in (with_dynamics(pendulum()), with_dynamics(fivebar_joint()), with_dynamics(scenario.ur5e_space())):
        for shaper in ("ZV", "ZVD"):
            shaped = input_shaped_move(space, mode_hz=3.0, damping_ratio=0.05, shaper=shaper)
            check_kind(shaped, space)
    space = with_dynamics(pendulum())
    start = space.home
    target = start + 0.8 * (space.room(start).high - start)
    law = TimeLaw.minimum_jerk(peak_speed_fraction=0.5)
    plain = law.leg(start, target, space.limits, space.rate_hz)
    for shaper in ("ZV", "ZVD"):
        shaped = input_shaped_move(space, target, mode_hz=3.0, damping_ratio=0.05, shaper=shaper)
        reference = shaped.positions[:, 0]
        end = shaped.t[-1]
        pad = np.arange(len(plain.t), len(shaped.t)) / space.rate_hz
        unshaped = np.concatenate([plain.q[:, 0], np.full(len(shaped.t) - len(plain.t), plain.q[-1, 0])])
        move = abs(target[0] - start[0])
        after = plain.t[-1]
        left_shaped = lane_c.residual_vibration(shaped.t, reference, 3.0, 0.05, after_s=end - 1e-9)
        # the response after the reference has stopped: the unshaped leg rings, the shaped one does not
        ring_plain = lane_c.residual_vibration(shaped.t, unshaped, 3.0, 0.05, after_s=after + 0.02) / move
        ring_shaped = lane_c.residual_vibration(shaped.t, reference, 3.0, 0.05, after_s=shaped.t[-1] - 0.0) / move
        assert pad[0] > 0 and left_shaped >= 0
        assert ring_plain > 0.01, f"the unshaped move should ring ({ring_plain:.4f})"
        assert ring_shaped < 0.02 * ring_plain + 1e-6, f"{shaper}: residual {ring_shaped:.5f} of the move vs {ring_plain:.4f} unshaped"
    # a wrong mode: ZVD is the less sensitive one (the point of the extra impulse), at 20 % error in the frequency
    off = {}
    for shaper in ("ZV", "ZVD"):
        shaped = input_shaped_move(space, target, mode_hz=3.0, damping_ratio=0.05, shaper=shaper)
        longer = np.concatenate([shaped.positions[:, 0], np.full(2000, shaped.positions[-1, 0])])
        t = np.arange(len(longer)) / space.rate_hz
        off[shaper] = lane_c.residual_vibration(t, longer, 3.6, 0.05, after_s=shaped.t[-1] + 0.01)
    assert off["ZVD"] < off["ZV"], off
    plain_peak = np.max(np.abs(plain.qd)), np.max(np.abs(plain.qdd))
    shaped = input_shaped_move(space, target, mode_hz=3.0, shaper="ZVD")
    assert np.max(np.abs(shaped.velocities)) <= plain_peak[0] + 1e-9 and np.max(np.abs(shaped.accelerations)) <= plain_peak[1] + 1e-9
    refused(lambda: input_shaped_move(pendulum()), "dynamics model")
    refused(lambda: input_shaped_move(space, shaper="EI"), "shaper")
    refused(lambda: input_shaped_move(space, mode_hz=-1.0), "mode_hz")


# ---------------------------------------------------------------------- cartesian_raster


def test_cartesian_raster_covers_the_reachable_cells_in_rows_at_the_tool_speeds():
    space, start = fivebar_reach(), FIVEBAR_START
    raster = cartesian_raster(space, tool_speeds_m_s=(0.05, 0.2), start=start)
    check_kind(raster, space)
    reachable_fivebar(raster)
    assert tracks(space, raster) <= MM
    points = space.tool_points(raster)
    # coverage: of the 20 mm cells that hold at least three reach-map tool points (clearly inside the reachable plane), the share
    # the tool enters; the reach map's own points are the independent oracle (the kind reads them only as a validity check)
    # WHY only the start's branch (the sign of the passive joint j3): a joint-space-continuous tool path stays on one branch
    branch = np.sign(robots.fivebar_passive(*FIVEBAR_START)[0])
    reach_points = np.array([space.forward(q)[:3, 3][:2] for q in space.reach.reachable
                             if (passive := robots.fivebar_passive(*q)) is not None and np.sign(passive[0]) == branch])
    cells, counts = np.unique(np.floor(reach_points / 0.02).astype(int), axis=0, return_counts=True)
    interior = {tuple(c) for c, n in zip(cells, counts) if n >= 3}
    entered = {tuple(c) for c in np.floor(points[:, :2] / 0.02).astype(int)}
    share = len(interior & entered) / len(interior)
    assert share >= 0.45, f"the raster enters {100 * share:.0f} % of the interior cells"
    # a row is straight along x: the rows of the first pass hold their y
    tool_all = space.tool_points(raster)

    def band_points(band) -> np.ndarray:
        return tool_all[(raster.t >= band.start_s - 1e-9) & (raster.t <= band.end_s + 1e-9)]

    rows = [band for band in bands(raster) if " row " in band.label]
    assert rows
    for band in rows:
        tool = band_points(band)
        assert np.ptp(tool[:, 1]) <= MM and np.ptp(tool[:, 2]) <= MM, "a row is not straight along x"
    # the tool speed: no faster than asked (3 %)
    for label_speed, asked in (("0.05 m/s", 0.05), ("0.2 m/s", 0.2)):
        speeds = [np.max(np.linalg.norm(np.diff(band_points(band), axis=0), axis=1) * space.rate_hz)
                  for band in bands(raster) if label_speed in band.label]
        assert speeds and max(speeds) <= asked * 1.03, (label_speed, max(speeds))
    assert any("pass 1 at 0.05 m/s: reached" in note for note in notes(raster))
    # the UR5e: a 10 cm square about the start, all 25 cells reachable, one row per cell row at the tool height
    arm = ur5e()
    square = cartesian_raster(arm, tool_speeds_m_s=(0.1,))
    check_kind(square, arm)
    assert tracks(arm, square) <= MM
    tool = arm.tool_points(square)
    assert np.ptp(tool[:, 2]) <= MM, "left the plane"
    assert any("5 rows over 5 x 5 cells" in note for note in notes(square)), notes(square)
    refused(lambda: cartesian_raster(arm, region_m=(5.0, 5.2, 5.0, 5.2)), "reachable")
    refused(lambda: cartesian_raster(arm, tool_speeds_m_s=(0.0,)), "positive")


# ---------------------------------------------------------------------- space_filling_pose_set

SOBOL_6D_FIRST_EIGHT = np.array([  # scipy.stats.qmc.Sobol(d=6, scramble=False).random(8): the independent oracle
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.75, 0.25, 0.25, 0.25, 0.75, 0.75],
    [0.25, 0.75, 0.75, 0.75, 0.25, 0.25], [0.375, 0.375, 0.625, 0.875, 0.375, 0.125], [0.875, 0.875, 0.125, 0.375, 0.875, 0.625],
    [0.625, 0.125, 0.875, 0.625, 0.625, 0.875], [0.125, 0.625, 0.375, 0.125, 0.125, 0.375]])


def stations_of(trajectory) -> np.ndarray:
    return np.array(segment_info(trajectory)[0]["stations"])


def test_space_filling_pose_set_is_stratified_and_a_pose_set():
    assert np.array_equal(_lane_c.sobol(8, 6), SOBOL_6D_FIRST_EIGHT), "the Sobol numbers are not scipy's"
    for method in ("sobol", "halton", "latin_hypercube"):
        for space in (pendulum(), fivebar_joint(), scenario.ur5e_space()):
            count = 32
            poses = space_filling_pose_set(space, count, method=method, seed=7, dwell_s=0.1, margin_fraction=0.0)
            check_kind(poses, space)
            stations = stations_of(poses)
            assert stations.shape == (count, space.dof)
            box = space.room(space.home)
            unit = (stations - box.low) / (box.high - box.low)
            assert np.all(unit >= 0) and np.all(unit <= 1)
            if method != "halton":  # one point in each of the count strata, in every dimension
                for joint in range(space.dof):
                    assert sorted(np.floor(unit[:, joint] * count).astype(int)) == list(range(count)), (method, joint)
            else:  # Halton: its base-2 dimension is stratified, the rest are balanced to within a few points per tenth
                assert sorted(np.floor(unit[:, 0] * count).astype(int)) == list(range(count))
                for joint in range(space.dof):
                    assert np.max(np.abs(np.histogram(unit[:, joint], bins=4, range=(0, 1))[0] - count / 4)) <= 3
            assert [band.label for band in bands(poses) if "station" in band.label] == [f"space-filling pose set: station {i}" for i in range(1, count + 1)]
            # every station is dwelled at: the first sample of its dwell is the station
            for index, station in enumerate(stations, start=1):
                row = np.flatnonzero(np.asarray(poses.channels["piece"]) == 2 * index - 1)[0]
                assert np.allclose(poses.positions[row], station, atol=1e-12)
    space = scenario.ur5e_space()
    a = space_filling_pose_set(space, 16, seed=1)
    assert not np.array_equal(stations_of(a), stations_of(space_filling_pose_set(space, 16, seed=2))), "the seed does not scramble"
    assert np.array_equal(stations_of(a), stations_of(space_filling_pose_set(space, 16, seed=1)))
    # a Sobol set beats an iid random one on coverage: the largest empty hole (distance to the nearest station, over a probe grid)
    unit_points = _lane_c.sobol(64, 2, 5)
    random_points = np.random.default_rng(5).random((64, 2))
    probe = np.random.default_rng(9).random((4000, 2))

    def hole(points: np.ndarray) -> float:
        return float(np.max(np.min(np.linalg.norm(probe[:, None] - points[None], axis=2), axis=1)))

    assert hole(unit_points) < hole(random_points), (hole(unit_points), hole(random_points))
    # collision: no station collides, and the legs between them are clear
    scene = space.with_collision(robots.UR5eScene())
    safe = space_filling_pose_set(scene, 16, seed=1)
    check_kind(safe, scene)
    assert not any(scene.in_collision(q) for q in stations_of(safe))
    refused(lambda: space_filling_pose_set(space, 8, method="sobol_scrambled"), "method")
    refused(lambda: space_filling_pose_set(space, 0), "count")
    refused(lambda: _lane_c.sobol(8, 11), "up to 10")


# ---------------------------------------------------------------------- lissajous_multi_joint


def test_lissajous_fills_the_room_with_incommensurate_sines():
    for space in (pendulum(), fivebar_joint(), scenario.ur5e_space()):
        curve = lissajous_multi_joint(space, duration_s=60.0)
        check_kind(curve, space)
        assert np.allclose(curve.positions[0], space.home) and np.allclose(curve.positions[-1], space.home), "does not return to its start"
    space = scenario.ur5e_space()
    curve = lissajous_multi_joint(space, duration_s=60.0)
    for a, b in ((0, 1), (1, 2), (0, 2), (3, 4)):  # a 2-D projection of the joint-space curve, 12 x 12 cells over its own extent
        x, y = curve.positions[::10, a], curve.positions[::10, b]
        cells = np.unique(np.floor(np.column_stack([(x - x.min()) / np.ptp(x), (y - y.min()) / np.ptp(y)]).clip(0, 0.999) * 12).astype(int), axis=0)
        assert len(cells) >= 0.55 * 144, f"joints {a},{b} visit {len(cells)} of 144 cells"
    single = lissajous_multi_joint(space, joints=["shoulder_pan_joint", "shoulder_lift_joint"], frequencies_hz=(0.13, 0.17), duration_s=60.0)
    assert np.allclose(single.positions[:, 2:], space.home[2:]), "joints not named moved"
    refused(lambda: lissajous_multi_joint(space, frequencies_hz=(0.1, 0.2, 0.3)), "commensurate")
    refused(lambda: lissajous_multi_joint(space, duration_s=3.0, fade_s=2.0), "fade_s")
    refused(lambda: lissajous_multi_joint(space, amplitude_fraction=1.5), "amplitude_fraction")


# ---------------------------------------------------------------------- coverage_greedy_swings


def occupancy_share(space, joint: int, trajectory, acceleration: bool) -> float:
    """An independent count of the cells the OUTPUT visits (every 5th sample), over the reachable ones: sim2real's cell rule."""
    room = space.room(space.home)
    low, high, vmax, amax = room.low[joint], room.high[joint], room.max_velocity[joint], room.max_acceleration[joint]
    qd_edges = np.array([-f * vmax for f in reversed((0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))] + [f * vmax for f in (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)])
    qdd_edges = np.array([-f * amax for f in reversed((0.00625, 0.025, 0.0625, 0.1875, 0.5, 1.0))] + [f * amax for f in (0.00625, 0.025, 0.0625, 0.1875, 0.5, 1.0)])
    q_edges = np.linspace(low, high, 13)
    samples = [trajectory.positions[::5, joint], trajectory.velocities[::5, joint]] + ([trajectory.accelerations[::5, joint]] if acceleration else [])
    edges = [q_edges, qd_edges] + ([qdd_edges] if acceleration else [])
    visited, _ = np.histogramdd(np.column_stack(samples), bins=edges)
    centres = np.meshgrid(*(0.5 * (e[:-1] + e[1:]) for e in edges), indexing="ij")
    room_to_stop = np.where(centres[1] > 0, high - centres[0], centres[0] - low)
    reachable = centres[1] ** 2 / (2 * amax) <= room_to_stop
    return 100.0 * float(np.count_nonzero((visited > 0) & reachable)) / float(np.count_nonzero(reachable))


def test_coverage_greedy_swings_fills_cells_greedily_to_the_target():
    for space, joint in ((pendulum(), "can11"), (fivebar_joint(), "j2"), (scenario.ur5e_space(), "elbow_joint")):
        swings = coverage_greedy_swings(space, joint, target_coverage_pct=25.0)
        check_kind(swings, space)
        index = space.joint_index(joint)
        others = np.delete(np.arange(space.dof), index)
        assert np.allclose(swings.positions[:, others], space.home[others]), "another joint moved"
        # the cells the output itself visits clear the target (transitions only add), counted independently of the kind
        share = occupancy_share(space, index, swings, True)
        assert share >= 25.0 - 1.0, f"{joint}: the output visits {share:.1f} % of the reachable cells"
    space = pendulum()
    low = coverage_greedy_swings(space, "can11", target_coverage_pct=10.0)
    high = coverage_greedy_swings(space, "can11", target_coverage_pct=40.0)
    assert high.t[-1] > low.t[-1] and len([b for b in bands(high) if "Hz" in b.label]) > len([b for b in bands(low) if "Hz" in b.label]), \
        "a higher target needs more swings"
    assert occupancy_share(space, 0, high, True) >= 38.0
    flat = coverage_greedy_swings(space, "can11", target_coverage_pct=40.0, acceleration_cells=False)
    check_kind(flat, space)
    assert occupancy_share(space, 0, flat, False) >= 38.0
    short = coverage_greedy_swings(space, "can11", target_coverage_pct=99.0, max_duration_s=20.0)
    assert any("falls short" in note for note in notes(short)), "an unreachable target is reported, not hidden"
    refused(lambda: coverage_greedy_swings(fivebar_joint()), "which joint")


# ---------------------------------------------------------------------- every kind, called as f(space)


def test_every_lane_c_kind_is_callable_with_the_space_alone():
    spaces = {"pendulum": pendulum(), "five-bar": fivebar_joint(), "UR5e": scenario.ur5e_space()}
    with_fk = {"five-bar": (fivebar(), FIVEBAR_START), "UR5e": (ur5e(), None)}
    for name in ("trapezoidal_velocity", "polynomial_move", "via_point_spline", "space_filling_pose_set", "lissajous_multi_joint"):
        for robot, space in spaces.items():
            kinds()[name].function(space)
    for name in ("cartesian_arc", "approach_retreat", "cartesian_a_to_b", "cartesian_raster"):
        for robot, (space, start) in with_fk.items():
            check_kind(kinds()[name].function(space, start=start), space)
            try:
                kinds()[name].function(spaces[robot])
            except MissingRung as error:
                assert "with_kinematics" in str(error)
            else:
                raise AssertionError(f"{name} generated without kinematics")
    for robot, space in spaces.items():
        kinds()["input_shaped_move"].function(with_dynamics(space))
        kinds()["coverage_greedy_swings"].function(space, space.joint_names[0])


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_lane_c (point-to-point + coverage)"))
