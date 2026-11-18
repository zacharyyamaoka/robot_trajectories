"""The trajectory catalog: every trajectory family worth supporting in the planned JointTrajectoryGenerator, one record each.

Zach (Daily Note, 2026-11-18): "I want like a library or Catalog of all the different trajectories". Swept 2026-11-18 from three
sources: our code (bam_ws, sim2real, robot_controllers, robot_msgs; every ``exists_in`` is ``path:line``), the vault (his notes from
2024-03 on repeatability, the Oct 29 test-schedule note, the Nov 18 brief) and primary prior art (papers by DOI, standards, the
tools that ship these signals: Rhoban BAM, MATLAB idinput / frest, ToddlerBot, FloBaRoID, BIRDy, Ruckig, TOPP-RA).

Rules every record keeps (``test_catalog.py`` holds them): a unique snake_case id, tags only from ``kind.py``'s vocabularies, at least
one use case, and either code we have (a path that exists) or a named primary source. De-duplicated: a family that differs from another
only by a parameter is one kind with that parameter (BAM's sin(t^2) is ``linear_chirp`` at constant angle; ISO 230-2's bidirectional
positioning is ``staircase``'s prior art). A ladder over a condition (speed, kp, payload, controller mode, seed) is a factor the
generator sweeps over a kind, not a kind, except ``speed_ladder_a_to_b``, which is a named protocol (the pendulum campaign's contract).

Display in degrees, data in SI: parameters here are rad, rad/s, s, Hz, m, N.m; a "fraction" is of the joint's own limit.

Paths vs trajectories (2026-11-18): every record says what it ``starts_as``, a time signal or a path, with a ``# WHY`` above each
borderline call; a Cartesian record names its geometry's ``dimensions``. ``defined_in``, joint-space ``dimensions`` and
``entry_level`` are derived (kind.py has every rule, and the citations).
"""

from __future__ import annotations

from .kind import TrajectoryKind

# ---------------------------------------------------------------------------------------------------------------- use cases

USE_CASES: dict[str, str] = {
    "pid_tuning": "choose or check feedback gains (kp, kd, ki) on a joint",
    "bandwidth": "closed-loop frequency response: bandwidth, phase lag, margins, resonances",
    "delay_id": "command and sensing delay (the phase a chirp or multisine reads at high frequency)",
    "inertia_id": "inertia / armature: the torque that goes with acceleration",
    "friction_id": "Coulomb, viscous and Stribeck friction, breakaway, cogging",
    "gravity_id": "gravity torque: masses, centres of mass, mount tilt",
    "backlash_id": "gear backlash, lost motion and transmission compliance",
    "full_dynamics_id": "the rigid-body base parameters of a multi-joint chain, all at once",
    "tracking_eval": "target vs sim vs real tracking error along the motion",
    "stop_accuracy": "settled error at a stop (Zach's 1 mm tool gate)",
    "state_coverage": "(q, qd, qdd) cell coverage of a joint: does the data visit where the model must be right",
    "workspace_coverage": "how much of the reachable workspace (tool positions and orientations) the motion visits",
    "repeatability": "the same input, run again: the floor every sim-vs-real gate is a multiple of",
    "accuracy": "absolute pose accuracy against an external truth",
    "kinematic_calibration": "link lengths, joint zeros and offsets from measured poses",
    "hand_eye_calibration": "camera-to-robot (or marker-to-flange) transforms",
    "camera_calibration": "camera intrinsics and distortion",
    "homing": "finding a joint's zero (hard stops) and getting back to READY from any rest",
    "thermal": "winding and board temperature: the thermal model and limits",
    "endurance": "wear, drift and faults over long running (burn-in, life)",
    "speed_limits": "the speed / acceleration envelope: how fast the joint or tool can go and still track",
    "disturbance_rejection": "recovery from an external push while holding",
    "contact_eval": "touching a surface: peak force, bounces, press-force error",
    "smoke_test": "a quick does-it-work-at-all pass before longer runs",
    "collision_check": "the motion's swept volume against the robot and the scene",
    "cycle_time": "how long a task cycle takes (picks per minute)",
    "demo": "motion that shows the robot off, or shows a person what it can do",
    "conveyor_pick_and_place": "moving between points that move (a conveyor's): intercept, track at the target's speed, leave",
}

# ---------------------------------------------------------------------------------------------------------------- code paths

RACK = "src/dev/bam_rack_twin/bam_rack_twin"
SCHEDULE = f"{RACK}/schedule.py"
EXPERIMENTS = f"{RACK}/hardware/experiments.py"
PID = "src/core/mdp/setup/bam_pid_tunning/src/bam_pid_tunning"
FIVEBAR = "src/dev/bam_fivebar/bam_fivebar"
TRAJGEN = "src/core/mdp/agent/actor/trajectory_generation"
REACH_DATASET = f"{TRAJGEN}/bam_reach/src/bam_reach/trajectory_dataset.py"
WRIST = "src/dev/bam_ur_wrist_coverage/bam_ur_wrist_coverage"
SIM2REAL = "/home/bam/sim2real/src/sim2real"
ROUTINES = f"{SIM2REAL}/cards/calibration.py"
RUNGS = "/home/bam/robot_controllers/src/robot_controllers/bench/rungs.py"
ROBOT_MSGS = "/home/bam/robot_msgs/src/robot_msgs"

# ---------------------------------------------------------------------------------------------------------------- prior art

BAM_TRAJ = ("Rhoban BAM, bam/trajectory.py: lift_and_drop, sin_time_square, up_and_down, sin_sin, half_sine, steps, nothing (read 2026-11-18)",
            "https://github.com/Rhoban/bam/blob/main/bam/trajectory.py")
BAM_PAPER = ("Duclusaud et al. 2024, Extended friction models for the physics simulation of servo actuators (BAM)", "https://arxiv.org/abs/2410.08650")
BAM_J288 = ("Rhoban BAM PR #21, Unitree J288 bam/unitree/all_record.py: torque_chirp 0.25 G, torque_steps 0.5 G, speed_sine, lift_and_damp (read 2026-11-18)",
            "https://github.com/Rhoban/bam/pull/21")
PINTELON = ("Pintelon & Schoukens 2012, System Identification: A Frequency Domain Approach, 2nd ed.", "https://doi.org/10.1002/9781118287422")
FARINA = ("Farina 2000, AES 108th Convention paper 5093, exponential swept sine", "https://aes2.org/publications/elibrary-page/?id=10211")
SCHROEDER = ("Schroeder 1970, IEEE Trans. Inf. Theory 16(1), low-peak-factor multisine phases", "https://doi.org/10.1109/TIT.1970.1054411")
IDINPUT = ("MATLAB System Identification Toolbox idinput: rgs, rbs, prbs, sine", "https://www.mathworks.com/help/ident/ref/idinput.html")
FREST_CHIRP = ("MATLAB Simulink Control Design frest.Chirp (linear / logarithmic / quadratic sweep)", "https://www.mathworks.com/help/slcontrol/ug/frest.chirp.html")
FREST_SINESTREAM = ("MATLAB Simulink Control Design frest.Sinestream (ramp, settle, measure per frequency)",
                    "https://www.mathworks.com/help/slcontrol/ug/frest.sinestream.html")
FREST_PRBS = ("MATLAB Simulink Control Design frest.PRBS", "https://www.mathworks.com/help/slcontrol/ug/frest.prbs.html")
PID_AUTOTUNER = ("MATLAB Closed-Loop PID Autotuner: sinusoids at 1/10, 1/3, 1, 3, 10 x the target bandwidth",
                 "https://www.mathworks.com/help/slcontrol/ug/closedlooppidautotuner.html")
TODDLERBOT = ("ToddlerBot sysID.py: linear chirp 0.1-10 over 10 s, 3 amplitudes x a kp sweep",
              "https://github.com/hshi74/toddlerbot/blob/main/toddlerbot/policies/sysID.py")
BECKHOFF = ("Beckhoff TE5960 TwinCAT Autotuning, Bode settings: Sine, MultiSine, Chirp, PRBS",
            "https://infosys.beckhoff.com/content/1033/te5960_tc3_autotuning/16674433419.html")
SWEVERS = ("Swevers, Ganseman, Tukel, De Schutter, Van Brussel 1997, IEEE T-RA, Optimal robot excitation and identification",
           "https://doi.org/10.1109/70.631234")
GAUTIER = ("Gautier & Khalil 1992, IJRR 11(4), Exciting trajectories for the identification of base inertial parameters of robots",
           "https://doi.org/10.1177/027836499201100408")
ARMSTRONG_1989 = ("Armstrong 1989, IJRR 8(6), On finding exciting trajectories for identification experiments", "https://doi.org/10.1177/027836498900800603")
CALAFIORE = ("Calafiore, Indri, Bona 2001, J. Robotic Systems 18(2), optimal excitation trajectories",
             "https://doi.org/10.1002/1097-4563(200102)18:2%3C55::AID-ROB1005%3E3.0.CO;2-O")
FLOBAROID = ("FloBaRoID: Fourier-series excitation optimised for D-optimality under joint and collision limits", "https://github.com/kjyv/FloBaRoID")
BIRDY = ("BIRDy (TUM-ICS): generateExcitationTrajectory.m, an optimised Fourier excitation benchmark", "https://github.com/TUM-ICS/BIRDy")
MORELLI_1999 = ("Morelli 1999, J. Aircraft, Flight test of optimal inputs and comparison with conventional inputs (doublet, 3-2-1-1)",
                "https://doi.org/10.2514/2.2469")
MORELLI_2003 = ("Morelli 2003, IFAC, Multiple input design for real-time parameter estimation (orthogonal multisines)",
                "https://doi.org/10.1016/S1474-6670(17)34833-4")
MORELLI_NTRS = ("Morelli 2021, NASA NTRS, Practical multiple-input design for aircraft system identification flight tests",
                "https://ntrs.nasa.gov/citations/20210018281")
ASTROM_MURRAY = ("Astrom & Murray, Feedback Systems, 2nd ed. (step response: rise time, overshoot, settling time)", "http://fbswiki.org/wiki/index.php/Main_Page")
ASTROM_HAGGLUND = ("Astrom & Hagglund 1984, Automatica 20(5), relay-feedback automatic tuning", "https://doi.org/10.1016/0005-1098(84)90014-1")
ARDUPILOT = ("ArduPilot Copter AutoTune: twitch inputs per axis", "https://ardupilot.org/copter/docs/autotune.html")
PX4 = ("PX4 multicopter autotune: a toggling step sequence on the rate setpoint", "https://docs.px4.io/main/en/config/autotune_mc.html")
ARMSTRONG_1994 = ("Armstrong-Helouvry, Dupont, Canudas de Wit 1994, Automatica 30(7), friction survey (constant-velocity runs, Stribeck)",
                  "https://doi.org/10.1016/0005-1098(94)90209-7")
LUGRE = ("Canudas de Wit, Olsson, Astrom, Lischinsky 1995, IEEE TAC 40(3), the LuGre friction model", "https://doi.org/10.1109/9.376053")
ODRIVE_COGGING = ("ODrive anticogging calibration: a slow one-way velocity ramp, then the other way", "https://docs.odriverobotics.com/v/latest/guides/anticogging.html")
NORDIN = ("Nordin & Gutman 2002, Automatica 38(10), Controlling mechanical systems with backlash: a survey", "https://doi.org/10.1016/S0005-1098(02)00047-X")
MA_HOLLERBACH = ("Ma & Hollerbach 1996, ICRA, Identifying mass parameters for gravity compensation", "https://users.cs.utah.edu/~jmh/Papers/Ma_ICRA96.pdf")
WERNHOLT = ("Wernholt 2004, Linkoping licentiate thesis 1131 (a square-wave offset under the broadband signal)",
            "https://rt.isy.liu.se/research/reports/LicentiateThesis/Lic1131.pdf")
FLASH_HOGAN = ("Flash & Hogan 1985, J. Neuroscience 5(7), the minimum-jerk model", "https://doi.org/10.1523/JNEUROSCI.05-07-01688.1985")
RUCKIG = ("Berscheid & Kroger 2021, RSS, Jerk-limited real-time trajectory generation with arbitrary target states (Ruckig)",
          "https://doi.org/10.15607/RSS.2021.XVII.015")
RUCKIG_REPO = ("Ruckig", "https://github.com/pantor/ruckig")
MODERN_ROBOTICS = ("Lynch & Park, Modern Robotics, ch. 9: trapezoidal, S-curve, cubic and quintic time scaling",
                   "https://hades.mech.northwestern.edu/images/2/25/MR-v2.pdf")
BIAGIOTTI = ("Biagiotti & Melchiorri 2008, Trajectory Planning for Automatic Machines and Robots (double-S, B-splines)",
             "https://doi.org/10.1007/978-3-540-85629-0")
SICILIANO = ("Siciliano et al. 2009, Robotics: Modelling, Planning and Control (via-point interpolation)", "https://doi.org/10.1007/978-1-84628-642-1")
ROS_JTC = ("ROS 2 joint_trajectory_controller, trajectory representation (linear / cubic / quintic spline; replacing a running trajectory)",
           "https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html")
RTB = ("Robotics Toolbox for Python: jtraj, mtraj, ctraj, mstraj, trapezoidal, quintic",
       "https://petercorke.github.io/robotics-toolbox-python/arm_trajectory.html")
TOPPRA = ("Pham & Pham 2018, IEEE T-RO 34(3), TOPP-RA", "https://doi.org/10.1109/TRO.2018.2819195")
TOPPRA_REPO = ("toppra", "https://github.com/hungpham2511/toppra")
SINGER = ("Singer & Seering 1990, ASME J. Dyn. Sys. Meas. Control 112(1), Preshaping command inputs to reduce system vibration",
          "https://doi.org/10.1115/1.2894142")
CHOSET = ("Choset 2000, Autonomous Robots 9(3), Coverage of known spaces: the boustrophedon cellular decomposition", "https://doi.org/10.1023/A:1008958800904")
KAVRAKI = ("Kavraki, Svestka, Latombe, Overmars 1996, IEEE T-RA 12(4), PRM: uniform random configuration sampling", "https://doi.org/10.1109/70.508439")
SOBOL = ("Sobol 1967, USSR Comput. Math. Math. Phys., the Sobol low-discrepancy sequence", "https://doi.org/10.1016/0041-5553(67)90144-9")
HALTON = ("Halton 1960, Numerische Mathematik, the Halton sequence", "https://doi.org/10.1007/BF01386213")
MCKAY = ("McKay, Beckman, Conover 1979, Technometrics, Latin hypercube sampling", "https://doi.org/10.1080/00401706.1979.10489755")
SCIPY_QMC = ("scipy.stats.qmc.Sobol", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html")
ZACHARIAS = ("Zacharias, Borst, Hirzinger 2007, IROS, Capturing robot workspace structure (the reachability map)", "https://doi.org/10.1109/IROS.2007.4399105")
BAZAEI = ("Bazaei, Yong, Moheimani 2012, Rev. Sci. Instrum. 83, Lissajous-scan pattern planning (AFM)", "https://doi.org/10.1063/1.4725525")
ISO_9283 = ("ISO 9283:1998, Manipulating industrial robots: performance criteria and related test methods (iso.org page not opened)",
            "https://www.iso.org/standard/22244.html")
ISO_9283_LISTING = ("ISO 9283:1998 scope listing (pose accuracy and repeatability, stabilisation time, overshoot, path accuracy, cornering)",
                    "https://standict.eu/standards-repository/manipulating-industrial-robots-performance-criteria-and-related-test-methods")
ISO_230_2 = ("ISO 230-2:2014, positioning accuracy and repeatability of NC axes (bidirectional)", "https://www.iso.org/standard/55295.html")
ISO_230_4 = ("ISO 230-4:2022, circular tests for NC machine tools", "https://www.iso.org/standard/79155.html")
R15_05 = ("ANSI/RIA R15.05-1, point-to-point and static performance characteristics of industrial robots",
          "https://webstore.ansi.org/standards/pcc/ansiriar15051992r1999")
BORM_MENQ = ("Borm & Menq 1991, IJRR 10(1), optimal measurement configurations by an observability measure", "https://doi.org/10.1177/027836499101000106")
SUN_HOLLERBACH = ("Sun & Hollerbach 2008, ICRA, Observability index selection for robot calibration", "https://doi.org/10.1109/ROBOT.2008.4543308")
HOLLERBACH_WAMPLER = ("Hollerbach & Wampler 1996, IJRR 15(6), The calibration index and taxonomy for robot kinematic calibration",
                      "https://doi.org/10.1177/027836499601500604")
TSAI_LENZ = ("Tsai & Lenz 1989, IEEE T-RA 5(3), fully autonomous and efficient 3D robotics hand/eye calibration", "https://doi.org/10.1109/70.34770")
PARK_MARTIN = ("Park & Martin 1994, IEEE T-RA, Robot sensor calibration: solving AX = XB on the Euclidean group", "https://doi.org/10.1109/70.326576")
ZHANG = ("Zhang 2000, IEEE TPAMI 22(11), A flexible new technique for camera calibration", "https://doi.org/10.1109/34.888718")
IEC_60034 = ("IEC 60034-1 rotating electrical machines, duty types S1-S10 (preview)", "https://webstore.iec.ch/p-preview/info_iec60034-1%7Bed12.0%7Db.pdf")
ADEPT = ("Omron/Adept iX4 manual: the Adept cycle, 25 mm up, 305 mm across, 25 mm down", "https://www.manualslib.com/manual/2043213/Omron-Ix4.html?page=155")
SCHNEIDER = ("Schneider Lexium Cobot typical cycle time (Adept cycle, arrival within 0.25 mm)",
             "https://product-help.schneider-electric.com/Lexium%20Cobot/V1.7/en/LXMCOBOT/LXMCOBOT/TPC_COBOT_TypicalCycleTime-74330732.html")
YCB = ("Calli et al. 2015, IEEE RAM, The YCB object and model set: benchmarking in manipulation research", "https://doi.org/10.1109/MRA.2015.2448951")
KIMBLE = ("Kimble et al. 2020, IEEE RA-L 5(2), Benchmarking protocols for evaluating small parts robotic assembly systems", "https://doi.org/10.1109/LRA.2020.2965869")
BRISTOW = ("Bristow, Tharayil, Alleyne 2006, IEEE Control Systems Magazine 26(3), A survey of iterative learning control", "https://doi.org/10.1109/MCS.2006.1636313")
MOTEUS_PID = ("moteus PID tuning guide: hold a position and disturb it by hand", "https://github.com/mjbots/moteus/blob/main/docs/guides/pid-tuning.md")
HWANGBO = ("Hwangbo et al. 2019, Science Robotics, actuator-net data from deployment-like sine foot motions (1-25 Hz)", "https://arxiv.org/abs/1901.08652")
PILZ = ("MoveIt 2 Pilz industrial motion planner: PTP (trapezoidal joint profiles, synchronised), LIN and CIRC (trapezoidal in "
        "Cartesian space; an arc by centre or interim point)",
        "https://moveit.picknik.ai/main/doc/how_to_guides/pilz_industrial_motion_planner/pilz_industrial_motion_planner.html")
MOVEIT_CARTESIAN = ("MoveIt 2 Move Group interface tutorial: pose goals, joint-space goals and computeCartesianPath (waypoints "
                    "interpolated at eef_step, IK at every step)",
                    "https://moveit.picknik.ai/main/doc/examples/move_group_interface/move_group_interface_tutorial.html")
CONVEYOR = "src/core/mdp/agent/actor/policy/bam_grasp/src/bam_grasp/conveyor.py"
ISAAC_SYSID = ("Isaac Sim system identification: sine sweeps, multisines, splines, and a separate validation trajectory",
               "https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_robot_setup_sysid.html")

# ---------------------------------------------------------------------------------------------------------------- the catalog

LIMITS = ("joint_limits", "velocity_limits", "acceleration_limits")
TIMED = (*LIMITS, "jerk_limits")

CATALOG: tuple[TrajectoryKind, ...] = (
    # ============================================================ excitation
    TrajectoryKind(
        id="linear_chirp", name="Linear chirp", family="excitation",
        one_line="A sine whose frequency rises linearly from f_start to f_end, its amplitude held at a constant speed (angle = v / 2 pi f) "
                 "or a constant angle (BAM's sin(t^2)), tapered in and out.",
        use_cases=("inertia_id", "bandwidth", "delay_id", "pid_tuning", "state_coverage"),
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("f_start_hz", "Hz", 0.5), ("f_end_hz", "Hz", 2.0), ("duration_s", "s", 20.0), ("speed_amplitude_fraction", "1", 0.8),
                    ("amplitude_rule", "text", "constant_speed"), ("taper_s", "s", 0.5), ("hold_after_s", "s", 1.0)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{EXPERIMENTS}:1174", f"{PID}/tapes.py:131", f"{SCHEDULE}:388", f"{SCHEDULE}:574"),
        prior_art=(PINTELON, FREST_CHIRP, TODDLERBOT, BAM_TRAJ),
        priority="now",
        notes="PID tuna's identification chirp stops at command_rate / 50 (2 Hz at a 100 Hz host): between commands moteus integrates a "
              "constant velocity, so a faster chirp breaks J qdd = tau (measured 2026-10-25: 5.7 Hz read a JMC's inertia +50 %, 2 Hz -3 %). "
              "BAM's sin(t^2) is this kind from 0 Hz at constant angle (registered as schedule's sin_time_square).",
    ),
    TrajectoryKind(
        id="log_chirp", name="Logarithmic chirp", family="excitation",
        one_line="An exponential sweep f0 -> f1 (up or down); at each frequency the amplitude is the smallest of the range, speed and "
                 "acceleration caps, so it passes through a pendulum's natural mode without leaving the envelope.",
        use_cases=("inertia_id", "bandwidth", "delay_id", "state_coverage"),
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("f0_hz", "Hz", 0.3), ("f1_hz", "Hz", 8.0), ("sweep_s", "s", 20.0), ("direction", "text", "up"),
                    ("range_fraction", "1", 0.85), ("speed_fraction", "1", 0.75)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{SCHEDULE}:359",),
        prior_art=(FARINA, PINTELON, FREST_CHIRP, BECKHOFF),
        priority="next",
        notes="Equal time per octave, so the low band (gravity, friction) and the mode (~3 Hz on CAN 11) both get energy. Above ~1 Hz the "
              "acceleration cap sets the amplitude. Stretch lowers the caps, never the band (the band is the point).",
    ),
    TrajectoryKind(
        id="slow_plus_fast_sines", name="Slow sine with a fast ripple", family="excitation",
        one_line="A slow large sine with a small fast sine on top: friction that depends on load while moving, and inertia separated "
                 "from gravity.",
        use_cases=("friction_id", "inertia_id", "state_coverage"),
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("slow_hz", "Hz", 0.15), ("fast_hz", "Hz", 2.5), ("slow_range_fraction", "1", 0.8), ("fast_to_slow_ratio", "1", 0.16),
                    ("duration_s", "s", 20.0), ("fade_s", "s", 1.0)),
        segments_join="periodic_returns_to_start",
        exists_in=(f"{SCHEDULE}:402",),
        prior_art=(BAM_TRAJ, BAM_PAPER),
        priority="next",
        notes="BAM's sin_sin is sin(t) pi/2 + 0.5 sin(5t) sin(2t) over 6 s; ours is the note's F4 rescaled to the joint's range.",
    ),
    TrajectoryKind(
        id="faded_sine", name="Faded sine", family="excitation",
        one_line="One sine at a fixed frequency and amplitude (optionally about an offset), faded in and out by raised cosines so it "
                 "starts and ends at rest where it began.",
        use_cases=("tracking_eval", "speed_limits", "bandwidth", "state_coverage"),
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("freq_hz", "Hz", 1.0), ("amplitude_rad", "rad", 0.2), ("cycles", "count", 5), ("ramp_cycles", "count", 1),
                    ("offset_rad", "rad", 0.0), ("command_mode", "text", "position")),
        segments_join="periodic_returns_to_start",
        exists_in=(f"{EXPERIMENTS}:1136", f"{RACK}/trajectory_dataset.py:69"),
        prior_art=(BAM_TRAJ, BAM_J288, HWANGBO),
        priority="next",
        notes="The 10-24 envelope ladder ran these at rising amplitude and frequency to find the speed and acceleration envelope (the fade "
              "exists because a plain sine starts with a velocity step the PD's kd kicks at). BAM's half_sine is one slow cycle; the J288 PR's "
              "speed_sine is this kind in velocity mode (kp = 0). trajectory_dataset's 'sweep' class is one faded sine at a random amplitude "
              "and frequency.",
    ),
    TrajectoryKind(
        id="stepped_sine_dwell", name="Stepped sine (sine dwell)", family="excitation",
        one_line="A list of frequencies played one after another: ramp in, settle a few cycles, measure amplitude and phase over a few "
                 "more; the classic point-by-point Bode measurement.",
        use_cases=("bandwidth", "pid_tuning", "delay_id"),
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("f_low_hz", "Hz", 0.3), ("f_high_hz", "Hz", 8.0), ("frequencies", "count", 12), ("settle_cycles", "count", 2),
                    ("measure_cycles", "count", 4), ("ramp_cycles", "count", 1), ("amplitude_rule", "text", "constant_speed")),
        segments_join="periodic_returns_to_start",
        exists_in=(f"{EXPERIMENTS}:1136",),
        prior_art=(FREST_SINESTREAM, PINTELON, BECKHOFF),
        priority="next",
        notes="Our code has the single dwell (the runner's faded 'sine' kind with cycles and ramp_cycles); the frequency list is not packaged "
              "as one kind yet. Slowest per point of all the frequency-response kinds and the most robust to nonlinearity.",
    ),
    TrajectoryKind(
        id="multisine", name="Periodic multisine", family="excitation",
        one_line="A periodic sum of sines on a fixed frequency grid (odd lines, every third skipped) with Schroeder phases and equal speed "
                 "per line, repeated for several periods.",
        use_cases=("bandwidth", "inertia_id", "delay_id", "state_coverage"),
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("period_s", "s", 10.0), ("periods", "count", 5), ("discard_periods", "count", 1), ("band_low_hz", "Hz", 0.3),
                    ("band_high_hz", "Hz", 6.0), ("velocity_rms_rad_s", "rad/s", 0.8), ("phase_rule", "text", "schroeder"),
                    ("lines", "text", "odd_skip_every_third")),
        segments_join="periodic_returns_to_start",
        exists_in=(f"{SCHEDULE}:474", f"{SCHEDULE}:467"),
        prior_art=(SCHROEDER, PINTELON, IDINPUT, PID_AUTOTUNER, SWEVERS),
        priority="next",
        notes="Periodicity gives a noise floor from the period-to-period spread; the skipped odd lines detect nonlinearity (Pintelon-Schoukens). "
              "idinput's 'sine' type is this kind with the lowest-crest of several random phase sets. The MATLAB closed-loop PID autotuner's "
              "five sines at 1/10..10 x the target bandwidth is this kind on a sparse grid, added at the plant input while holding.",
    ),
    TrajectoryKind(
        id="orthogonal_multisine", name="Orthogonal multisine (every joint at once)", family="excitation",
        one_line="Every joint gets its own disjoint set of harmonics of one fundamental, so all joints move together and each joint's "
                 "response still separates in frequency.",
        use_cases=("full_dynamics_id", "bandwidth", "state_coverage"),
        starts_as="time_signal",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=(*LIMITS, "collision_model"),
        parameters=(("period_s", "s", 10.0), ("periods", "count", 5), ("band_low_hz", "Hz", 0.3), ("band_high_hz", "Hz", 4.0),
                    ("velocity_rms_rad_s", "rad/s", 0.5)),
        segments_join="periodic_returns_to_start",
        exists_in=(),
        prior_art=(MORELLI_2003, MORELLI_NTRS),
        priority="later",
        notes="Not in our code. The multi-joint answer to Zach's 11-11 'I don't see a lot of multi-joint motion and coverage' "
              "(bam_ur_wrist_coverage multi.py) that keeps per-joint frequency separation.",
    ),
    TrajectoryKind(
        id="prbs", name="Pseudo-random binary sequence (PRBS)", family="excitation",
        one_line="A maximum-length binary sequence (period 2^n - 1) of two levels, each held one clock period, rate-limited so a position "
                 "loop can follow it.",
        use_cases=("bandwidth", "delay_id", "inertia_id"),
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("register_stages", "count", 7), ("clock_period_s", "s", 0.05), ("amplitude_rad", "rad", 0.05), ("periods", "count", 2)),
        segments_join="periodic_returns_to_start",
        exists_in=(),
        prior_art=(IDINPUT, FREST_PRBS, BECKHOFF, PX4),
        priority="later",
        notes="Flat spectrum to about 0.44 / clock period. Binary position steps are harsh on a position-controlled joint: drives inject it on "
              "the velocity or current loop. PX4's autotune toggles 0 / +A with shrinking holds, a cousin.",
    ),
    TrajectoryKind(
        id="band_limited_noise", name="Band-limited random noise", family="excitation",
        one_line="Gaussian noise filtered to a band and scaled so speed and acceleration stay inside the limits; a new realisation per seed.",
        use_cases=("bandwidth", "inertia_id", "state_coverage"),
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("band_low_hz", "Hz", 0.1), ("band_high_hz", "Hz", 4.0), ("rms_rad", "rad", 0.05), ("duration_s", "s", 30.0), ("seed", "count", 0)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(),
        prior_art=(IDINPUT, PINTELON),
        priority="later",
        notes="idinput's 'rgs'. Needs windowing or averaging across realisations, where a periodic multisine needs neither.",
    ),
    TrajectoryKind(
        id="fourier_series_excitation", name="Optimised Fourier-series excitation", family="excitation",
        one_line="Each joint a finite Fourier series of one fundamental, its coefficients optimised under position, speed, acceleration and "
                 "collision limits to make the dynamics regressor well-conditioned.",
        use_cases=("full_dynamics_id", "gravity_id", "inertia_id", "friction_id", "state_coverage"),
        # WHY time_signal: the coefficients are optimised for the regressor at these frequencies: re-timing undoes the optimisation
        starts_as="time_signal",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=(*LIMITS, "dynamics_model", "collision_model"),
        parameters=(("fundamental_hz", "Hz", 0.1), ("harmonics", "count", 5), ("periods", "count", 4), ("criterion", "text", "condition_number")),
        segments_join="periodic_returns_to_start",
        exists_in=(f"{ROUTINES}:548",),
        prior_art=(SWEVERS, GAUTIER, ARMSTRONG_1989, CALAFIORE, FLOBAROID, BIRDY),
        priority="next",
        notes="Declared as the sim2real routine arm.dynamics_excitation ('Fourier excitations and the coverage sweep'); no optimiser in our code. "
              "FloBaRoID (Python, D-optimality, collision constraints) and BIRDy (MATLAB) are the stock parts. Periodic, so it averages like "
              "the multisine.",
    ),
    TrajectoryKind(
        id="biased_dither", name="Biased dither (slide without reversing)", family="excitation",
        one_line="A slow constant-speed ramp across the range carrying a small multisine whose speed never exceeds the ramp's, so the joint "
                 "slides without reversing: inertia and viscous friction with no stick-slip.",
        use_cases=("inertia_id", "friction_id"),
        # WHY time_signal: never reversing is a speed relation in time (the ripple slower than the ramp), not a shape in space
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("ramp_speed_fraction", "1", 0.075), ("lines_low_hz", "Hz", 1.0), ("lines_high_hz", "Hz", 6.0),
                    ("ripple_speed_fraction_of_ramp", "1", 0.8), ("range_fraction", "1", 0.85)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{SCHEDULE}:420",),
        prior_art=(WERNHOLT,),
        priority="next",
    ),
    TrajectoryKind(
        id="torque_chirp", name="Open-loop torque chirp", family="excitation",
        one_line="With kp = kd = 0, a gravity hold plus a linear torque chirp: the plant with nothing else in the loop.",
        use_cases=("inertia_id", "friction_id", "delay_id", "bandwidth"),
        starts_as="time_signal",
        space="torque", scope="single_joint", output="torque_profile", needs=("joint_limits", "torque_limits", "dynamics_model"),
        parameters=(("f0_hz", "Hz", 0.5), ("f1_hz", "Hz", 6.0), ("duration_s", "s", 15.0), ("torque_fraction_of_gravity", "1", 0.25)),
        segments_join="ends_anywhere",
        exists_in=(f"{SCHEDULE}:536",),
        prior_art=(BAM_J288, PINTELON),
        priority="next",
        notes="A pendulum only (gravity brings it back); amplitude min(0.25 G, G sin(0.85 x range)) keeps the static deflection in the window. "
              "Registered runner=False: needs Zach. Confirmed 2026-11-18 in the J288 PR's all_record.py (0.25 G 'because the chirp sweeps "
              "through the pendulum's natural frequency').",
    ),
    TrajectoryKind(
        id="torque_steps", name="Open-loop torque steps", family="excitation",
        one_line="Open-loop torque steps (kp = kd = 0) on top of a gravity hold: torque-to-acceleration gain and the friction band with no "
                 "position loop.",
        use_cases=("inertia_id", "friction_id"),
        starts_as="time_signal",
        space="torque", scope="single_joint", output="torque_profile", needs=("joint_limits", "torque_limits", "dynamics_model"),
        parameters=(("step_torque_fraction_of_gravity", "1", 0.25), ("hold_s", "s", 1.0), ("steps", "count", 4)),
        segments_join="ends_anywhere",
        exists_in=(),
        prior_art=(BAM_J288,),
        priority="later",
        notes="Not in our code. J288's 0.5 G steps would leave CAN 11's window (Servo Test Schedule note, F9), so size it as torque_chirp is sized.",
    ),
    TrajectoryKind(
        id="free_swing_release", name="Free swing (lift and release)", family="excitation",
        one_line="Lift the joint under its PD, then release it (torque off, or kd only) and let it coast: inertia, damping and Coulomb "
                 "friction with no controller in the loop.",
        use_cases=("inertia_id", "friction_id", "gravity_id"),
        # WHY time_signal: no geometry to re-time: the content is torque off for coast_s and whatever physics does; the lift only sets it up
        starts_as="time_signal",
        space="torque", scope="single_joint", output="command_sequence", needs=("joint_limits", "velocity_limits"),
        parameters=(("release_fraction_of_side", "1", 0.6), ("lift_speed_fraction", "1", 0.1), ("coast_s", "s", 8.0), ("release_kd", "N.m.s/rad", 0.0)),
        segments_join="ends_anywhere",
        exists_in=(f"{SCHEDULE}:524", f"{EXPERIMENTS}:1110"),
        prior_art=(BAM_TRAJ, BAM_J288, BAM_PAPER),
        priority="next",
        notes="BAM's lift_and_drop releases after a 2 s cubic lift ('removes back-EMF', the paper); 'nothing' is the torque-off rest; J288's "
              "lift_and_damp keeps kd. It ends wherever it comes to rest, so the next segment starts from the measured pose.",
    ),
    # ============================================================ controller_test
    TrajectoryKind(
        id="step_train", name="Step train", family="controller_test",
        one_line="Short sharp moves into long holds, alternating sign, with two larger slower steps that cross the friction band: "
                 "overshoot, ringing and settling are read from it.",
        use_cases=("pid_tuning", "bandwidth", "stop_accuracy"),
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=("joint_limits", "velocity_limits"),
        parameters=(("count", "count", 6), ("sharp_move_command_periods", "count", 2), ("large_step_fraction_of_travel", "1", 0.15), ("hold_s", "s", 0.8)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{PID}/tapes.py:162", f"{PID}/tune.py:178"),
        prior_art=(ASTROM_MURRAY, ARDUPILOT),
        priority="now",
        notes="The bandwidth sweep (tune.py) raises the loop's natural frequency one guarded rung at a time on this tape until it rings: a "
              "gain ladder is a condition swept over a kind. Sharp because a 50 ms minimum-jerk move has no energy at a 100+ Hz loop's bandwidth.",
    ),
    TrajectoryKind(
        id="staircase", name="Staircase (bidirectional levels)", family="controller_test",
        one_line="Climb to +A in levels, descend to -A, climb back, holding each level, so every inner level is reached once from below and "
                 "once from above.",
        use_cases=("backlash_id", "stop_accuracy", "friction_id", "repeatability"),
        # WHY path: the levels and the side each is approached from are the content; move speed and hold length are timing choices (ISO
        #   230-2 runs it at any feed). Contrast step_train, whose sharp two-command-period moves ARE the input
        starts_as="path",
        space="joint", scope="single_joint", output="joint_trajectory", needs=("joint_limits", "velocity_limits"),
        parameters=(("levels", "count", 7), ("range_fraction", "1", 0.85), ("move_speed_fraction", "1", 0.1), ("hold_s", "s", 1.5)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{EXPERIMENTS}:1311", f"{SCHEDULE}:511", f"{RACK}/backlash.py:155", "src/dev/bam_rack_twin/docs/gate_plans_1029.py:108"),
        prior_art=(BAM_TRAJ, ISO_230_2, NORDIN),
        priority="next",
        notes="Backlash = the output-camera vs encoder offset between the two approach directions at a settled hold (backlash.py; needs a camera "
              "on the output). ISO 230-2's bidirectional positioning (targets from both sides, repeated cycles) is the machine-tool form; "
              "BAM's 'steps' is a one-sided staircase.",
    ),
    TrajectoryKind(
        id="doublet_3211", name="Doublet and 3-2-1-1 pulses", family="controller_test",
        one_line="A doublet (+A then -A) or the 3-2-1-1 train (3, 2, 1, 1 time units of alternating sign): a short input with a broad, "
                 "tunable spectrum.",
        use_cases=("bandwidth", "inertia_id", "pid_tuning"),
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("unit_time_s", "s", 0.2), ("amplitude_rad", "rad", 0.05), ("pattern", "text", "3-2-1-1")),
        segments_join="starts_and_ends_at_rest",
        exists_in=(),
        prior_art=(MORELLI_1999, MORELLI_NTRS),
        priority="later",
        notes="Flight-test inputs, used where time on the vehicle is expensive (as arm time is here). Pulse edges need the same rate limit as PRBS.",
    ),
    TrajectoryKind(
        id="relay_feedback", name="Relay feedback test", family="controller_test",
        one_line="Replace the controller by a relay (+d / -d torque on the sign of the error) and read the limit cycle's amplitude and "
                 "period: the ultimate gain and period a PID rule starts from.",
        use_cases=("pid_tuning", "bandwidth"),
        # WHY time_signal: a switching rule in time (the relay follows the measurement): no geometry
        starts_as="time_signal",
        space="torque", scope="single_joint", output="command_sequence", needs=("joint_limits", "torque_limits"),
        parameters=(("relay_torque_nm", "N.m", 0.2), ("hysteresis_rad", "rad", 0.001), ("cycles", "count", 6)),
        segments_join="ends_anywhere",
        exists_in=(),
        prior_art=(ASTROM_HAGGLUND,),
        priority="later",
        notes="Ku = 4 d / (pi a). Not a precomputed trajectory: the switching follows the measurement. Our tuner starts from an identified "
              "model instead (analytic gains, then the step-train sweep).",
    ),
    TrajectoryKind(
        id="rapid_replan", name="Rapid replanning", family="controller_test",
        one_line="A new trajectory every 0.15 s, each starting from the previous one's state at that instant (position and velocity), to "
                 "seeded targets near home.",
        use_cases=("tracking_eval", "smoke_test", "conveyor_pick_and_place"),
        # WHY time_signal: the 0.15 s replan clock is what it tests (state handed over mid-motion); the targets are incidental
        starts_as="time_signal",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=(*LIMITS, "home_pose"),
        parameters=(("replan_period_s", "s", 0.15), ("replans", "count", 8), ("target_spread_rad", "rad", 0.349), ("move_s", "s", 0.6), ("seed", "count", 0)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{RUNGS}:520", f"{RUNGS}:264"),
        prior_art=(ROS_JTC,),
        priority="later",
        notes="Starts and ends at rest; every inner segment joins mid-motion (state handed over), the one join kind no other entry exercises.",
    ),
    # ============================================================ quasi_static
    TrajectoryKind(
        id="velocity_plateaus", name="Constant-velocity plateaus", family="quasi_static",
        one_line="Constant-speed plateaus out and back at a ladder of speeds, each held long enough to settle: friction against speed "
                 "(Coulomb, viscous, Stribeck, breakaway).",
        use_cases=("friction_id", "speed_limits", "state_coverage"),
        # WHY time_signal: the plateaus ARE the measurement (friction at each held speed, each long enough to settle): any other time law
        #   over the same out-and-back erases them. Contrast speed_ladder_a_to_b, where the shape of each timing does not matter
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("speed_fractions", "text", "0.025,0.05,0.1,0.175,0.25,0.35"), ("plateau_s", "s", 2.0), ("min_plateau_s", "s", 0.8),
                    ("first_direction", "text", "+")),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{EXPERIMENTS}:1194", f"{SCHEDULE}:342", f"{PID}/tapes.py:131", f"{ROUTINES}:381", f"{ROUTINES}:605"),
        prior_art=(ARMSTRONG_1994, LUGRE, ODRIVE_COGGING),
        priority="now",
        notes="PID tuna's ident-sweeps cross the travel at 20-100 % of top speed both ways; sim2real's servo.top_speed_t7 is the top rung "
              "(drive ceilings); conveyor.dynamics_profiles fits belt mass and drag from this kind (speed steps and plateaus) on the belt axis. A speed whose plateau would be under 0.8 s is left to the "
              "sines. ODrive's anticogging is a slow velocity ramp both ways.",
    ),
    TrajectoryKind(
        id="constant_speed_triangle", name="Slow constant-speed triangle (gravity sweep)", family="quasi_static",
        one_line="A slow constant-speed triangle 0 -> +A -> -A -> 0 (or one-sided), corners rounded at a set acceleration: gravity torque, "
                 "Coulomb friction and the rotor-angle (cogging) map at every angle, both directions.",
        use_cases=("gravity_id", "friction_id", "backlash_id", "state_coverage"),
        # WHY path: it reads gravity and friction against ANGLE: the angles visited, both ways, are the content and any slow constant speed
        #   reads the same (unlike velocity_plateaus, where speed is the variable)
        starts_as="path",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("amplitude_fraction", "1", 0.95), ("speed_fraction", "1", 0.04), ("corner_accel_rad_s2", "rad/s^2", 1.0), ("one_sided", "count", 0)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{EXPERIMENTS}:1128", f"{SCHEDULE}:328", f"{RACK}/integral_study.py:82", f"{ROUTINES}:562"),
        prior_art=(BAM_TRAJ, ARMSTRONG_1994, ODRIVE_COGGING),
        priority="next",
        notes="Sweeping through the angle where gravity torque changes sign flips the load across the backlash (sim2real "
              "arm.backlash_gravity_sweep). integral_study.sweep_q rounds every speed change with a raised cosine of set peak accel. One-sided "
              "for a joint that rests on a stop (CAN 13). BAM's up_and_down is the cubic cousin with a static plateau.",
    ),
    TrajectoryKind(
        id="breakaway_torque_ramp", name="Breakaway torque ramp", family="quasi_static",
        one_line="Hold one flank, ramp the torque cap (or feed-forward) until the output leaves, both ways: static friction per direction.",
        use_cases=("friction_id", "backlash_id"),
        starts_as="time_signal",
        space="torque", scope="single_joint", output="command_sequence", needs=("joint_limits", "torque_limits"),
        parameters=(("ramp_rate_nm_s", "N.m/s", 0.5), ("start_torque_nm", "N.m", 0.0), ("leave_threshold_rad", "rad", 0.002)),
        segments_join="ends_anywhere",
        exists_in=(f"{ROUTINES}:484",),
        prior_art=(ARMSTRONG_1994,),
        priority="next",
    ),
    TrajectoryKind(
        id="small_reversals", name="Small slow reversals at a held pose", family="quasi_static",
        one_line="Small slow back-and-forth moves about a held pose (the load on one flank, then the other): backlash and lost motion, read "
                 "by an output-side camera or tracker.",
        use_cases=("backlash_id", "repeatability"),
        # WHY path: the reversal points either side of the held pose are the content; the speed only has to be slow
        starts_as="path",
        space="joint", scope="single_joint", output="joint_trajectory", needs=("joint_limits", "velocity_limits", "camera"),
        parameters=(("amplitude_rad", "rad", 0.01), ("speed_rad_s", "rad/s", 0.02), ("reversals", "count", 6), ("dwell_s", "s", 1.0)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{ROUTINES}:455", f"{ROUTINES}:399", f"{ROUTINES}:576"),
        prior_art=(NORDIN,),
        priority="next",
    ),
    TrajectoryKind(
        id="locked_output_torque_ramp", name="Locked-output torque ramp", family="quasi_static",
        one_line="With the output clamped in a fixture, ramp the motor torque up and down both ways and read the motor angle: backlash and "
                 "gear stiffness.",
        use_cases=("backlash_id",),
        starts_as="time_signal",
        space="torque", scope="single_joint", output="torque_profile", needs=("torque_limits",),
        parameters=(("peak_torque_nm", "N.m", 2.0), ("ramp_s", "s", 4.0), ("cycles", "count", 2)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{ROUTINES}:412",),
        prior_art=(NORDIN,),
        priority="later",
    ),
    TrajectoryKind(
        id="quasi_static_pose_holds", name="Dithered static pose holds", family="quasi_static",
        one_line="Hold a spread set of poses, dithering slowly in motor space so Coulomb friction cancels, and average the torques: masses, "
                 "centres of mass and mount tilt by regression.",
        use_cases=("gravity_id",),
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="pose_set",
        needs=("joint_limits", "forward_kinematics", "dynamics_model", "collision_model"),
        parameters=(("poses", "count", 48), ("hold_s", "s", 2.0), ("dither_amplitude_rad", "rad", 0.01), ("min_fold_margin", "1", 0.1), ("seed", "count", 20261113)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{FIVEBAR}/massid.py:617", f"{RACK}/hardware/full_arm.py:464", f"{ROUTINES}:526", f"{ROUTINES}:371"),
        prior_art=(MA_HOLLERBACH,),
        priority="next",
        notes="full_arm.comp_poses draws admitted bent-elbow five-bar poses (fold margin >= 0.1) as the comp's fit and check set; "
              "arm.gravity_direction reads the mount tilt from the same holds; servo.known_mass holds a weighed mass for the torque scale. "
              "The 11-14 lesson: holds on static friction over-fed the wrist (x1.22); the 15c fit used friction-cancelled MOVING rows.",
    ),
    # ============================================================ point_to_point
    TrajectoryKind(
        id="minimum_jerk_move", name="Minimum-jerk move", family="point_to_point",
        one_line="A rest-to-rest move x = x0 + d (10 s^3 - 15 s^4 + 6 s^5), s = t / T, peaking at 1.875 d / T: the house leg of every "
                 "waypoints tape.",
        use_cases=("tracking_eval", "stop_accuracy", "pid_tuning", "demo"),
        # WHY path: the move A -> B is the content. Minimum jerk, trapezoid, S-curve, polynomial and TOPP-RA are interchangeable time laws
        #   for it: these point-to-point kinds are the timing layer every path kind picks from
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=("joint_limits", "velocity_limits"),
        parameters=(("move_s", "s", 1.0), ("dwell_s", "s", 0.3)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{EXPERIMENTS}:1278", f"{RUNGS}:68", f"{FIVEBAR}/gains_check.py:219"),
        prior_art=(FLASH_HOGAN, MODERN_ROBOTICS),
        priority="now",
        notes="The runner's waypoints 'q:move_s:dwell_s;...' are minimum-jerk legs that return to the start; its speed check uses the 1.875 peak.",
    ),
    TrajectoryKind(
        id="joint_waypoints", name="Joint-space waypoints", family="point_to_point",
        one_line="Rest-to-rest moves through a list of absolute joint configurations, each leg timed by one time law, dwelling at "
                 "each: the classic PTP sequence of a 6-DOF arm, on any joint count.",
        use_cases=("tracking_eval", "stop_accuracy", "cycle_time", "demo"),
        # WHY path: the configurations are the content; the time law (minimum-jerk, trapezoid, a plugin's) is a choice
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("peak_speed_fraction", "1", 0.5), ("dwell_s", "s", 0.2)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{EXPERIMENTS}:1278",),
        prior_art=(PILZ, RTB, MODERN_ROBOTICS),
        priority="now",
        notes="Zach 11-18: 'What about like multi waypoint trajectories?' minimum_jerk_move takes fractions of the room around its "
              "start; this takes absolute configurations (a pose library, READY, a calibration set).",
    ),
    TrajectoryKind(
        id="trapezoidal_velocity", name="Trapezoidal velocity move", family="point_to_point",
        one_line="Accelerate at a, cruise at v, decelerate at a (a triangle when the distance is too short to reach v).",
        use_cases=("tracking_eval", "cycle_time"),
        # WHY path: a time law for a given move (see minimum_jerk_move)
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("velocity_fraction", "1", 1.0), ("acceleration_fraction", "1", 1.0)),
        segments_join="starts_and_ends_at_rest",
        exists_in=("src/dev/bam_grasp_worlds/bam_grasp_worlds/trajectory.py:57", f"{RACK}/hardware/fivebar_homed.py:1402"),
        prior_art=(MODERN_ROBOTICS, RTB),
        priority="later",
    ),
    TrajectoryKind(
        id="jerk_limited_scurve", name="Jerk-limited S-curve (Ruckig)", family="point_to_point",
        one_line="A time-optimal move to any target state under speed, acceleration and jerk limits (the double-S profile), joints "
                 "synchronised in time or in phase.",
        use_cases=("tracking_eval", "cycle_time", "speed_limits"),
        # WHY path: a time law for a given move (see minimum_jerk_move)
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=TIMED,
        parameters=(("velocity_fraction", "1", 1.0), ("acceleration_fraction", "1", 1.0), ("jerk_fraction", "1", 1.0), ("synchronization", "text", "phase")),
        segments_join="starts_and_ends_at_rest",
        exists_in=("src/dev/bam_trajgen_bench/bam_trajgen_bench/timelaw.py:34", f"{FIVEBAR}/tapes.py:232",
                   "src/dev/bam_grasp_worlds/bam_grasp_worlds/carry.py:157", f"{TRAJGEN}/bam_traj_gen/src/bam_traj_gen/ruckig_backend.py:74"),
        prior_art=(RUCKIG, RUCKIG_REPO, BIAGIOTTI),
        priority="next",
        notes="Phase synchronisation keeps a multi-joint move straight (memory five-bar-regression, 'Phase, not Time'). "
              "bam_trajgen_bench.timelaw is the exact closed form for one joint, held against Ruckig's recorded durations.",
    ),
    TrajectoryKind(
        id="polynomial_move", name="Cubic or quintic polynomial move", family="point_to_point",
        one_line="A cubic (position and velocity at both ends) or quintic (plus acceleration) polynomial between two states over a set duration.",
        use_cases=("tracking_eval",),
        # WHY path: a time law for a given move (see minimum_jerk_move)
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=("joint_limits", "velocity_limits"),
        parameters=(("order", "count", 5), ("duration_s", "s", 1.0)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{RUNGS}:78",),
        prior_art=(MODERN_ROBOTICS, SICILIANO, RTB, BAM_TRAJ),
        priority="later",
        notes="Minimum jerk is the quintic with zero end velocity and acceleration; this kind lets the ends move (joins mid-motion). BAM's "
              "up_and_down and lift_and_drop are cubic keyframe splines ([t, x, dx/dt]).",
    ),
    TrajectoryKind(
        id="via_point_spline", name="Via-point spline", family="point_to_point",
        one_line="A smooth trajectory through a list of waypoints (spline or blended segments, stopping only where asked), time-parameterised "
                 "under the limits.",
        use_cases=("tracking_eval", "cycle_time", "demo"),
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=TIMED,
        parameters=(("interpolation", "text", "quintic_spline"), ("blend_radius_m", "m", 0.05), ("stop_at", "text", "first,last")),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{TRAJGEN}/bam_traj_gen/src/bam_traj_gen/traj_generator.py:546",),
        prior_art=(ROS_JTC, SICILIANO, RTB, RUCKIG),
        priority="next",
    ),
    TrajectoryKind(
        id="time_optimal_path", name="Time-optimal path parameterisation", family="point_to_point",
        one_line="Fix the geometric path, then find the fastest timing along it under speed and acceleration (and torque) limits.",
        use_cases=("cycle_time", "speed_limits"),
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("grid_points", "count", 200),),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{TRAJGEN}/bam_traj_gen/src/bam_traj_gen/toppra_utils.py:14",),
        prior_art=(TOPPRA, TOPPRA_REPO),
        priority="later",
        notes="The vault's 'Trajectory Controller Archaeology' note records why LERP survived Ruckig on the UR5e cell.",
    ),
    TrajectoryKind(
        id="cartesian_straight_line", name="Straight tool line", family="point_to_point",
        one_line="The tool moves on a straight line between two poses with a rest-to-rest time law, mapped to joints by IK at every sample "
                 "and refused (never clipped) if any sample leaves the envelope.",
        use_cases=("tracking_eval", "accuracy", "workspace_coverage"),
        starts_as="path", dimensions=("1D",),  # a line; the tool's orientation is held, not shaped
        space="cartesian", scope="cartesian_path", output="joint_trajectory",
        needs=(*TIMED, "reach_map", "forward_kinematics", "inverse_kinematics"),
        parameters=(("tool_speed_m_s", "m/s", 0.1), ("time_law", "text", "ruckig_phase")),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{FIVEBAR}/tapes.py:323", f"{RUNGS}:155"),
        prior_art=(RTB, ISO_9283),
        priority="next",
    ),
    TrajectoryKind(
        id="cartesian_waypoints", name="Cartesian waypoints", family="point_to_point",
        one_line="The tool visits a list of points or poses and dwells at each; inverse kinematics at the points only, joint-space "
                 "legs between them (the tool does not move straight between points).",
        use_cases=("tracking_eval", "stop_accuracy", "accuracy", "cycle_time", "demo"),
        starts_as="path", dimensions=("2D", "3D", "6D"),  # points in a plane (the five-bar), a volume, or full poses (the UR5e)
        space="cartesian", scope="coordinated_multi_joint", output="joint_trajectory",
        needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
        parameters=(("peak_speed_fraction", "1", 0.5), ("dwell_s", "s", 0.2)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{TRAJGEN}/bam_traj_gen/src/bam_traj_gen/traj_generator.py:546",),
        prior_art=(MOVEIT_CARTESIAN, PILZ),
        priority="next",
        notes="The Cartesian twin of joint_waypoints: MoveIt's pose goals in a row. cartesian_straight_line is the straight-line version.",
    ),
    TrajectoryKind(
        id="cartesian_arc", name="Circular arc of the tool (CIRC)", family="point_to_point",
        one_line="The tool moves on a circular arc from where it is to a goal, the arc set by its centre or by a point it passes "
                 "through, at rest at both ends, inverse kinematics at every step.",
        use_cases=("tracking_eval", "accuracy", "demo"),
        starts_as="path", dimensions=("2D",),  # an arc lies in a plane; its orientation is held or interpolated, not shaped
        space="cartesian", scope="cartesian_path", output="joint_trajectory",
        needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
        parameters=(("interim_or_centre", "text", "interim"), ("peak_speed_fraction", "1", 0.5)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(),
        prior_art=(PILZ, ISO_9283),
        priority="next",
        notes="Pilz's CIRC. circular_test is the full closed circle measured as a ballbar; this is the arc move of the 6-DOF family.",
    ),
    TrajectoryKind(
        id="approach_retreat", name="Approach and retreat along the tool axis", family="point_to_point",
        one_line="A straight tool move along its own axis to a pose and back out (or out only): the short linear legs before and "
                 "after a point, held to the tool's orientation.",
        use_cases=("tracking_eval", "stop_accuracy", "cycle_time", "contact_eval"),
        starts_as="path", dimensions=("1D",),  # a line along the tool axis
        space="cartesian", scope="cartesian_path", output="joint_trajectory",
        needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
        parameters=(("distance_m", "m", 0.05), ("peak_speed_fraction", "1", 0.3), ("dwell_s", "s", 0.3)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{REACH_DATASET}:145",),
        prior_art=(PILZ, MOVEIT_CARTESIAN),
        priority="next",
        notes="bam_reach's pick_and_lift recipe is an approach plus a retreat; the axis is the tool's own z unless one is given.",
    ),
    TrajectoryKind(
        id="cartesian_a_to_b", name="Pose-to-pose move from the reach map", family="point_to_point",
        one_line="Sample a start and a goal pose (optionally a via pose) from the reach map's pose sources, solve IK at the endpoints and time "
                 "a joint-space move between them.",
        use_cases=("tracking_eval", "workspace_coverage", "cycle_time"),
        # two poses wherever the reach map reaches: the five-bar's plane, a volume, the UR5e's full poses
        starts_as="path", dimensions=("2D", "3D", "6D"),
        space="cartesian", scope="coordinated_multi_joint", output="joint_trajectory",
        needs=(*TIMED, "reach_map", "inverse_kinematics", "collision_model"),
        parameters=(("recipe", "text", "a_to_b"), ("n_intermediate_points", "count", 3), ("blend_radius_m", "m", 0.05), ("seed", "count", 0)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{REACH_DATASET}:49", f"{REACH_DATASET}:545"),
        prior_art=(ZACHARIAS, RUCKIG),
        priority="next",
        notes="The 'I think I had something like this already' of the Nov 18 note: bam_reach.trajectory_dataset composes ReachMap pose "
              "sources with bam_traj_gen recipes (a_to_b, a_via_c_to_b, pick_and_lift, pick_and_place) and keeps failures as cases.",
    ),
    TrajectoryKind(
        id="input_shaped_move", name="Input-shaped move", family="point_to_point",
        one_line="A move convolved with an impulse sequence (ZV or ZVD) tuned to a lightly damped mode, so it ends with no residual vibration.",
        use_cases=("tracking_eval", "stop_accuracy", "cycle_time"),
        # WHY time_signal: it goes A -> B like the other moves, but the impulse sequence timed to the mode IS the point: re-timing it brings
        #   the vibration back
        starts_as="time_signal",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=("joint_limits", "velocity_limits", "dynamics_model"),
        parameters=(("mode_hz", "Hz", 3.0), ("damping_ratio", "1", 0.05), ("shaper", "text", "ZVD")),
        segments_join="starts_and_ends_at_rest",
        exists_in=(),
        prior_art=(SINGER,),
        priority="later",
    ),
    # ============================================================ coverage
    TrajectoryKind(
        id="joint_grid_snake", name="Joint-space grid snake", family="coverage",
        one_line="A boustrophedon tour over a grid of joint levels (the inner joint reverses each row), rest-to-rest moves between "
                 "waypoints; the held-out tour is an offset grid in shuffled order.",
        use_cases=("state_coverage", "workspace_coverage", "tracking_eval", "stop_accuracy"),
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=(*TIMED, "collision_model"),
        parameters=(("levels_per_joint", "count", 4), ("held_out", "text", "offset_grid_shuffled"), ("dwell_s", "s", 0.5)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{WRIST}/multi3.py:139", f"{WRIST}/multi3.py:165", f"{WRIST}/dwell3.py:48"),
        prior_art=(CHOSET,),
        priority="next",
    ),
    TrajectoryKind(
        id="cartesian_raster", name="Workspace raster sweep", family="coverage",
        one_line="The reachable plane cut into square cells, one straight tool row per cell row, boustrophedon, so the tool crosses every "
                 "reachable cell; timed at a ladder of tool speeds.",
        use_cases=("workspace_coverage", "tracking_eval", "speed_limits"),
        starts_as="path", dimensions=("2D",),  # a plane cut into cells
        space="cartesian", scope="cartesian_path", output="joint_trajectory",
        needs=(*TIMED, "reach_map", "forward_kinematics", "inverse_kinematics", "collision_model"),
        parameters=(("cell_m", "m", 0.02), ("tool_speeds_m_s", "text", "0.05,0.1,0.2,0.35,0.6,1.0"), ("ramp_time_s", "s", 0.25), ("jerk_time_s", "s", 0.1)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{FIVEBAR}/tapes.py:182", f"{FIVEBAR}/tapes.py:376", f"{FIVEBAR}/speed_ladder.py:267"),
        prior_art=(CHOSET, ZACHARIAS),
        priority="next",
        notes="Zach's 'workspace sweep'. The five-bar's real speed ladder runs it vertical and tilted, gravity FF and full FF.",
    ),
    TrajectoryKind(
        id="random_targets", name="Random reachable targets", family="coverage",
        one_line="Seeded uniformly random reachable configurations visited one after another by rest-to-rest moves; another seed is the "
                 "held-out set.",
        use_cases=("state_coverage", "workspace_coverage", "tracking_eval", "stop_accuracy", "smoke_test"),
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=("joint_limits", "velocity_limits", "collision_model"),
        parameters=(("targets", "count", 8), ("seed", "count", 0), ("speed_fraction_low", "1", 0.3), ("speed_fraction_high", "1", 0.8), ("dwell_s", "s", 0.5)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{FIVEBAR}/study.py:46", f"{TRAJGEN}/bam_reach/src/bam_reach/sampling.py:22",
                   "src/core/mdp/setup/bam_calibration/src/bam_calibration/rig.py:154"),
        prior_art=(KAVRAKI,),
        priority="next",
        notes="Zach, 2026-11-18: 'send it random commands and lets see the performance'. For one joint this is deployment_motion_classes' "
              "pickplace; this kind is its multi-joint form.",
    ),
    TrajectoryKind(
        id="space_filling_pose_set", name="Space-filling pose set (Sobol, Halton, Latin hypercube)", family="coverage",
        one_line="Poses from a low-discrepancy (Sobol, Halton) or stratified (Latin hypercube) sequence over the joint box or the reach map, "
                 "so N poses cover the space evenly with no clusters.",
        use_cases=("workspace_coverage", "kinematic_calibration", "gravity_id", "homing"),
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="pose_set", needs=("joint_limits", "collision_model"),
        parameters=(("method", "text", "sobol"), ("count", "count", 64), ("scramble_seed", "count", 0)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{SIM2REAL}/sim/homing/bench.py:71", f"{SIM2REAL}/sim/homing/fullarm_check.py:296"),
        prior_art=(SOBOL, HALTON, MCKAY, SCIPY_QMC),
        priority="next",
        notes="sim2real's homing bench draws its start states from a Sobol sequence. Sobol wants N = 2^m points.",
    ),
    TrajectoryKind(
        id="lissajous_multi_joint", name="Lissajous (incommensurate sines per joint)", family="coverage",
        one_line="Each joint a sine at its own frequency, the frequencies near-incommensurate, so the joint-space curve fills the box densely "
                 "with smooth periodic motion.",
        use_cases=("state_coverage", "workspace_coverage", "full_dynamics_id"),
        # WHY path: written as sines, but the frequency RATIOS set a curve that fills the box; any re-timing keeps the curve and its
        #   coverage
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=(*LIMITS, "collision_model"),
        parameters=(("frequencies_hz", "text", "0.13,0.17,0.23"), ("amplitude_fraction", "1", 0.8), ("duration_s", "s", 60.0)),
        segments_join="periodic_returns_to_start",
        exists_in=(),
        prior_art=(BAZAEI,),
        priority="later",
        notes="Only scan-pattern prior art found (AFM), no robot-identification paper. A Lissajous curve is a one-line-per-joint multisine, so "
              "it inherits that analysis; it is also the prettiest landing-page demo of coverage.",
    ),
    TrajectoryKind(
        id="coverage_greedy_swings", name="Coverage-picked tapered swings", family="coverage",
        one_line="Tapered sine swings, one per (centre, amplitude, frequency), picked greedily by how many empty (q, qd, qdd) cells each adds "
                 "until the predicted coverage clears a target.",
        use_cases=("state_coverage", "inertia_id", "friction_id"),
        # WHY time_signal: picked by (q, qd, qdd) cell coverage, so its speeds and accelerations are the content, not only the angles
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=(*LIMITS, "collision_model"),
        parameters=(("target_coverage_pct", "1", 40.0), ("candidates", "text", "centre x amplitude x frequency")),
        segments_join="periodic_returns_to_start",
        exists_in=(f"{FIVEBAR}/coverage_set.py:144", f"{FIVEBAR}/coverage_set.py:68", f"{SIM2REAL}/schedule/joint.py:454"),
        prior_art=(ISAAC_SYSID,),
        priority="next",
        notes="The only kind chosen BY the coverage metric (sim2real.schedule.joint.occupancy): Zach's 11-04 gate, 'bin the space and check "
              "that X % of cells are covered'. On the five-bar crank it lifted C1 from 34.1 % toward the proposed 40 %.",
    ),
    # ============================================================ standard_test
    TrajectoryKind(
        id="repeat_floor_tape", name="Repeat-floor tape", family="standard_test",
        one_line="One frozen tape played N times with the same input: the run-to-run floor every sim-vs-real gate is a multiple of.",
        use_cases=("repeatability", "tracking_eval"),
        # WHY time_signal: the point is the identical input, timing included, played N times (the tape it repeats may itself be a path)
        starts_as="time_signal",
        space="joint", scope="single_joint", output="joint_trajectory", needs=("joint_limits", "velocity_limits"),
        parameters=(("repeats", "count", 5), ("tape", "text", "the station's floor tape")),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{SCHEDULE}:320", f"{RACK}/hardware/tape.py:231"),
        prior_art=(ISO_9283, BRISTOW),
        priority="now",
        notes="Pendulum contract: REPEATS = 5 per rung; sim vs real must be <= 1.5x the same-input repeat floor (pendulum_goal.py on lane "
              "claude/1dof-rack-sim-to-real-435fc0-v4, not on main). Repeating one trajectory is also what iterative learning control feeds on.",
    ),
    TrajectoryKind(
        id="speed_ladder_a_to_b", name="A-to-B speed ladder", family="standard_test",
        one_line="Minimum-jerk A -> B -> A moves whose peak speed is a rising fraction of the joint's top speed, each rung repeated, plus "
                 "held-out A/B pairs: tracking and stop accuracy as speed rises to the motor's limit.",
        use_cases=("tracking_eval", "stop_accuracy", "speed_limits", "repeatability", "pid_tuning"),
        # WHY path: one path (A -> B -> A) timed at a ladder of peak speeds: the path-velocity decomposition in its purest form
        starts_as="path",
        space="joint", scope="single_joint", output="joint_trajectory", needs=LIMITS,
        parameters=(("rung_fractions", "text", "0.2,0.4,0.6,0.8,0.9,1.0"), ("repeats", "count", 5), ("held_out", "count", 2),
                    ("shape", "text", "minimum_jerk")),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{RACK}/hardware/tape.py:231", f"{FIVEBAR}/speed_ladder.py:267", f"{WRIST}/plans.py:71"),
        prior_art=(ISO_9283, R15_05),
        priority="now",
        notes="The pendulum campaign's frozen contract (pendulum_goal.py RUNG_FRACTIONS / REPEATS / HELD_OUT, lane "
              "claude/1dof-rack-sim-to-real-435fc0-v4). The rack's 'tape' kind is a raised-cosine out-and-back per speed. ISO 9283 also runs "
              "its pose tests at several rated speeds.",
    ),
    TrajectoryKind(
        id="iso9283_pose_cube", name="ISO 9283 pose test (cube, P1-P5)", family="standard_test",
        one_line="Five poses on a diagonal plane of a cube in the workspace, visited in order for many cycles, each approached from the same "
                 "direction: pose accuracy, repeatability, stabilisation time and overshoot.",
        use_cases=("accuracy", "repeatability", "stop_accuracy"),
        # five poses on the cube's diagonal plane so all three axes move; ISO scores position and orientation
        starts_as="path", dimensions=("3D", "6D"),
        space="cartesian", scope="coordinated_multi_joint", output="pose_set",
        needs=("joint_limits", "velocity_limits", "reach_map", "forward_kinematics", "inverse_kinematics", "camera"),
        parameters=(("cube_edge_m", "m", 0.2), ("cycles", "count", 30), ("speed_fraction", "1", 1.0), ("dwell_s", "s", 2.0)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(),
        prior_art=(ISO_9283, ISO_9283_LISTING, R15_05),
        priority="next",
        notes="P1-P5 and 30 cycles are from memory of the standard, not re-read (paywalled). Our encoder-scored workspace sweeps (r1a-r1d, "
              "11-17) are the house version, scored by sim2real.metrics.actuator.stops; Zach's gate is 1 mm PER STOP, stricter than ISO's mean.",
    ),
    TrajectoryKind(
        id="iso9283_path", name="ISO 9283 path test (line, circle, corners)", family="standard_test",
        one_line="A straight line, a circle and a rectangle's corners traced at set tool speeds: path accuracy, path repeatability, cornering "
                 "deviation and speed fidelity.",
        use_cases=("tracking_eval", "accuracy", "repeatability"),
        starts_as="path", dimensions=("1D", "2D"),  # a line (1D), a circle and a rectangle's corners (2D)
        space="cartesian", scope="cartesian_path", output="joint_trajectory",
        needs=(*LIMITS, "reach_map", "forward_kinematics", "inverse_kinematics", "camera"),
        parameters=(("tool_speed_m_s", "m/s", 0.1), ("circle_diameter_m", "m", 0.1), ("cycles", "count", 10)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(),
        prior_art=(ISO_9283, ISO_9283_LISTING),
        priority="later",
    ),
    TrajectoryKind(
        id="circular_test", name="Circular test (ballbar)", family="standard_test",
        one_line="A circle traced by two axes together (two joints, or the tool in a plane), in both directions: circular error and the "
                 "spikes at each axis reversal.",
        use_cases=("tracking_eval", "backlash_id", "accuracy"),
        starts_as="path", dimensions=("2D",),  # a circle in a plane
        space="cartesian", scope="cartesian_path", output="joint_trajectory", needs=(*LIMITS, "reach_map", "inverse_kinematics"),
        parameters=(("diameter_m", "m", 0.1), ("feed_m_s", "m/s", 0.05), ("directions", "text", "both")),
        segments_join="periodic_returns_to_start",
        exists_in=(),
        prior_art=(ISO_230_4,),
        priority="later",
        notes="The quadrant reversals show backlash and friction as spikes, which makes it a cheap two-joint backlash screen.",
    ),
    # ============================================================ calibration
    TrajectoryKind(
        id="kinematic_calibration_pose_set", name="Kinematic calibration pose set", family="calibration",
        one_line="Static poses chosen to make the calibration's identification Jacobian well-conditioned (an observability index), each "
                 "measured by an external tracker or camera.",
        use_cases=("kinematic_calibration", "accuracy"),
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="pose_set",
        needs=("joint_limits", "forward_kinematics", "collision_model", "camera"),
        parameters=(("poses", "count", 30), ("criterion", "text", "observability_index"), ("candidates", "count", 2000)),
        segments_join="starts_and_ends_at_rest",
        exists_in=("src/core/mdp/setup/bam_calibration/src/bam_calibration/rig.py:154", f"{ROUTINES}:242"),
        prior_art=(BORM_MENQ, SUN_HOLLERBACH, HOLLERBACH_WAMPLER),
        priority="next",
        notes="bam_calibration samples leader configurations on the assembled five-bar branch and rejects branch-ambiguous ones (real data "
              "never rides the fold). Zach's 2024-03 notebook: system ID on ~20 parameters (lengths plus backlash), not thousands.",
    ),
    TrajectoryKind(
        id="hand_eye_station_set", name="Hand-eye station set", family="calibration",
        one_line="Stations spread over the whole workspace (left and right) with large rotations between them, a marker or board seen at each: "
                 "AX = XB hand-eye, then a held-out check-set to refit and validate.",
        use_cases=("hand_eye_calibration", "accuracy"),
        starts_as="path", dimensions=("6D",),  # large rotations between stations are the point
        space="cartesian", scope="coordinated_multi_joint", output="pose_set",
        needs=("joint_limits", "reach_map", "forward_kinematics", "inverse_kinematics", "collision_model", "camera"),
        parameters=(("stations", "count", 30), ("min_rotation_between_rad", "rad", 0.52), ("check_set_stations", "count", 15)),
        segments_join="starts_and_ends_at_rest",
        exists_in=("src/core/mdp/setup/bam_calibration/research/hand_eye_2026_11_06/hand_eye_study.py:284", f"{ROUTINES}:628"),
        prior_art=(TSAI_LENZ, PARK_MARTIN),
        priority="next",
        notes="Zach (DP - Hand Eye Calibration, 11-13): cover the whole workspace; accuracy plateaus after X poses. The 11-15 overhead D435 "
              "fit on 4 runs at one J1 angle overfit (0.88 px own, 4-9 px held out); the refit over J1 -5..-30 deg halved held-out error.",
    ),
    TrajectoryKind(
        id="camera_board_views", name="Calibration board views", family="calibration",
        one_line="A ChArUco board (carried by the arm, or fixed and seen from a camera the arm carries) at many tilts and positions across "
                 "the image: intrinsics and distortion.",
        use_cases=("camera_calibration",),
        starts_as="path", dimensions=("6D",),  # board tilts and positions across the image
        space="cartesian", scope="coordinated_multi_joint", output="pose_set", needs=("reach_map", "inverse_kinematics", "camera"),
        parameters=(("views", "count", 20), ("max_tilt_rad", "rad", 0.785), ("image_cells", "count", 9)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{ROUTINES}:647",),
        prior_art=(ZHANG,),
        priority="later",
    ),
    TrajectoryKind(
        id="hard_stop_seek", name="Hard-stop seek (homing)", family="calibration",
        one_line="One joint pushes toward its hard stop at a slow slewed setpoint under a low torque cap that steps up, the others held, until "
                 "it stalls at the top cap; that angle defines the zero (one touch, N touches averaged, or press-probe-re-press).",
        use_cases=("homing",),
        # WHY time_signal: no geometry to time: the end is what it finds; the content is the slewed setpoint and the stepped torque-cap
        #   schedule until a stall
        starts_as="time_signal",
        space="joint", scope="per_joint_sequential", output="command_sequence", needs=("velocity_limits", "torque_limits"),
        parameters=(("speed_rad_s", "rad/s", 0.1), ("cap_steps_nm", "text", "0.5,1.0,2.0"), ("stall_lead_rad", "rad", 0.02),
                    ("max_travel_rad", "rad", 0.5), ("touches", "count", 1)),
        segments_join="ends_anywhere",
        exists_in=(f"{FIVEBAR}/stop_seek.py:131", f"{RUNGS}:638", "src/core/mdp/setup/bam_homing/docs/ref/sensorless_homing.py:105",
                   f"{ROUTINES}:196", f"{ROUTINES}:215", f"{ROUTINES}:226"),
        prior_art=(),
        priority="next",
        notes="Zach's procedure (11-10): J2 to its stop, then C1, then J1. A stall at a low cap may be gravity or friction, so the cap steps up; "
              "a stall at the top cap is the stop wherever it is, never an assumed angle. Needs no limits: it is what finds them.",
    ),
    TrajectoryKind(
        id="return_to_ready", name="Return to READY", family="calibration",
        one_line="A JointCommandSequence that walks the arm from wherever it rests to READY along waypoints that keep fold margins, then "
                 "settles: the rig's recover-yourself move.",
        use_cases=("homing", "smoke_test"),
        # WHY path: the waypoints that keep fold margins are the content; the slow speed is a timing choice
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="command_sequence",
        needs=("joint_limits", "forward_kinematics", "collision_model", "home_pose"),
        parameters=(("waypoint_step_rad", "rad", 0.087), ("speed_rad_s", "rad/s", 0.1), ("settle_s", "s", 1.0)),
        segments_join="starts_and_ends_at_rest",
        exists_in=("src/dev/bam_fivebar_recovery/bam_fivebar_recovery/home.py:163", "src/dev/bam_fivebar_recovery/bam_fivebar_recovery/home.py:65",
                   f"{ROBOT_MSGS}/control/sequence_controller.py:68"),
        prior_art=(),
        priority="next",
        notes="Zach 11-11: recover a stuck arm yourself with the return-to-home JointCommandSequence. The past-straight fold escape (11-15) is "
              "a variant whose first step picks the escape direction.",
    ),
    # ============================================================ stress
    TrajectoryKind(
        id="burn_in_wiggle", name="Burn-in wiggle", family="stress",
        one_line="Each joint walked to its homed zero, then swept about it one at a time at rising speeds with the others held, then gripper "
                 "open/close cycles, every tick logged (current, bus voltage, board temperature, faults).",
        use_cases=("smoke_test", "endurance", "thermal"),
        # WHY path: a sweep about zero run at a ladder of speeds, as speed_ladder_a_to_b is
        starts_as="path",
        space="joint", scope="per_joint_sequential", output="joint_trajectory",
        needs=("joint_limits", "velocity_limits", "home_pose", "collision_model"),
        parameters=(("amplitude_rad", "rad", 0.436), ("speeds_rad_s", "text", "0.349,0.698,1.047"), ("sweeps", "count", 2), ("claw_cycles", "count", 3)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{RACK}/hardware/wrist_burnin.py:75", "src/dev/bam_rack_twin/tests/test_wrist_burnin_1113.py:44"),
        prior_art=(),
        priority="next",
        notes="Zach's step 3 (11-13): 'burn-in wiggle ... with no faults'. A PASS needs the joint to have tracked its sweep, every pose checked "
              "against the self-collision guard first. Defaults are 25 deg at 20 / 40 / 60 deg/s.",
    ),
    TrajectoryKind(
        id="heating_run", name="Winding heating run", family="stress",
        one_line="Hold the rotor at standstill with a constant d-axis current, read the winding temperature from its resistance between heats, "
                 "then log the cool-down: the motor's thermal model.",
        use_cases=("thermal",),
        starts_as="time_signal",
        space="torque", scope="single_joint", output="torque_profile", needs=("torque_limits",),
        parameters=(("heat_current_fraction", "1", 0.5), ("probe_every_s", "s", 10.0), ("heat_s", "s", 600.0), ("cool_s", "s", 900.0)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{RACK}/hardware/thermal.py:280", f"{ROUTINES}:390"),
        prior_art=(IEC_60034,),
        priority="later",
        notes="A current, not a motion (the rotor holds stepper-fashion); its temperatures are provisional until checked against a thermometer.",
    ),
    TrajectoryKind(
        id="endurance_duty_cycle", name="Endurance duty cycle", family="stress",
        one_line="A deployment cycle (or a duty-type profile) repeated for hours with logging and periodic gap checks: wear, drift and thermal "
                 "steady state.",
        use_cases=("endurance", "thermal", "repeatability"),
        # WHY time_signal: the duty profile in time (IEC 60034 S1-S10) is the content; which motion carries it is incidental
        starts_as="time_signal",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=("joint_limits", "velocity_limits", "home_pose"),
        parameters=(("cycles", "count", 10000), ("duty_type", "text", "S1"), ("gap_check_every_s", "s", 43200.0)),
        segments_join="periodic_returns_to_start",
        exists_in=(f"{ROUTINES}:425",),
        prior_art=(IEC_60034,),
        priority="later",
        notes="sim2real servo.gap_check_forecast: the deployed backlash check every 12 h on the daily motions. MOC - Testing Robotic Systems "
              "lists 'gripper cycle-forever'.",
    ),
    TrajectoryKind(
        id="power_up_random_rests", name="Power-up from random rests", family="stress",
        one_line="Start states drawn where a hand or a fall can leave the arm (singular and flipped rests oversampled), then home or recover "
                 "from each: robustness of the power-up path.",
        use_cases=("homing", "smoke_test"),
        starts_as="path",
        space="joint", scope="coordinated_multi_joint", output="pose_set", needs=("joint_limits", "forward_kinematics", "collision_model"),
        parameters=(("power_ups", "count", 320), ("seed", "count", 20261110), ("strata", "text", "singular,flipped,free")),
        segments_join="ends_anywhere",
        exists_in=(f"{FIVEBAR}/powerup_stress.py:145", f"{SIM2REAL}/sim/homing/fullarm_home.py:648"),
        prior_art=(KAVRAKI,),
        priority="later",
        notes="Sim only by design (the arm falls unpowered into its rest); the start states, not a trajectory, are the generated part.",
    ),
    # ============================================================ task_cycle
    TrajectoryKind(
        id="deployment_motion_classes", name="Deployment motion classes (pickplace, cycle, jog, sweep)", family="task_cycle",
        one_line="Seeded random tapes in four deployment classes: pickplace (large moves, long holds), cycle (one pair out and back x3), jog "
                 "(small corrections) and sweep (one slow pass); other seeds are the sealed held-out set.",
        use_cases=("tracking_eval", "stop_accuracy", "pid_tuning"),
        # WHY path: three of its four classes (pickplace, cycle, jog) are seeded minimum-jerk waypoint tapes, random_targets on one joint;
        #   the fourth (sweep) is a faded_sine
        starts_as="path",
        space="joint", scope="single_joint", output="joint_trajectory", needs=("joint_limits", "velocity_limits"),
        parameters=(("classes", "text", "pickplace,cycle,jog,sweep"), ("train_seeds", "text", "0,1,2"), ("held_out_seeds", "text", "3,4,5")),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{RACK}/trajectory_dataset.py:57", f"{PID}/tapes.py:188", f"{SCHEDULE}:300"),
        prior_art=(HWANGBO, ISAAC_SYSID),
        priority="now",
        notes="PID tuna scores every candidate gain set on these (a different seed is the held-out set, so a pick is never proven on the tape it "
              "was chosen on); the rack schedule's OPERATE set and H2 acceptance set are the same classes.",
    ),
    TrajectoryKind(
        id="pick_and_place_cycle", name="Pick-and-place cycle", family="task_cycle",
        one_line="Approach above, pre-grasp, a two-stage descent (hover, settle, slow final line), close, lift, carry, place, release, return: "
                 "the cycle the facility runs, scored in picks per minute.",
        use_cases=("cycle_time", "tracking_eval", "stop_accuracy", "contact_eval", "demo", "conveyor_pick_and_place"),
        starts_as="path", dimensions=("3D", "6D"),  # approach, descend, lift and carry in space; the grasp's orientation on a 6-DOF arm
        space="cartesian", scope="coordinated_multi_joint", output="joint_trajectory",
        needs=(*TIMED, "reach_map", "forward_kinematics", "inverse_kinematics", "collision_model", "camera"),
        parameters=(("above_z_offset_m", "m", 0.20), ("pre_grasp_z_offset_m", "m", 0.08), ("hover_above_top_m", "m", 0.015), ("final_descent_m_s", "m/s", 0.03)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{REACH_DATASET}:49", f"{TRAJGEN}/bam_traj_gen/src/bam_traj_gen/traj_generator.py:546",
                   "src/dev/bam_trajgen_bench/bam_trajgen_bench/objective.py:131", f"{RACK}/hardware/pick_stops.py:1"),
        prior_art=(YCB, KIMBLE),
        priority="next",
        notes="Zach 11-12: don't score traj gen on facility KPIs; keep its own (SPPM in bam_trajgen_bench). The two-stage descent is "
              "pick_stops' fix for cogging error on a long descent.",
    ),
    TrajectoryKind(
        id="moving_target_track", name="Approach and track a moving point", family="task_cycle",
        one_line="From rest, meet a point that moves (a pose as a function of time) at the sync time with its velocity matched, "
                 "then follow it within tolerance to the end of the sync window, then come to rest.",
        use_cases=("tracking_eval", "conveyor_pick_and_place", "cycle_time"),
        # WHY path: the target's motion is geometry given in time; the approach and leave legs take a time law
        starts_as="path", dimensions=("1D", "2D", "3D", "6D"),  # a point moving on a line, a plane, in space, or a full pose
        space="cartesian", scope="cartesian_path", output="joint_trajectory",
        needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
        parameters=(("sync_start_s", "s", 1.0), ("sync_end_s", "s", 2.0), ("tolerance_m", "m", 0.001)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{CONVEYOR}:72", f"{RUNGS}:336"),
        prior_art=(),
        priority="next",
        notes="Zach 11-18: 'You can account for like trajectory points that are moving'. Contract and ruler: "
              "robot_trajectories.moving (synchronised). bam_grasp's conveyor leads the target by 0.25 s to cancel servo lag.",
    ),
    TrajectoryKind(
        id="meet_moving_point", name="Meet a moving point, end moving with it", family="task_cycle",
        one_line="From rest, meet a point that moves at the sync time with its velocity matched and stay on it to the end of the "
                 "sync window, ending there in motion with it: the first half of a moving-target cycle (track_and_release is the "
                 "second).",
        use_cases=("tracking_eval", "conveyor_pick_and_place"),
        # WHY path: the target's motion is geometry given in time; the approach leg takes a time law (moving_target_track's call)
        starts_as="path", dimensions=("1D", "2D", "3D", "6D"),
        space="cartesian", scope="cartesian_path", output="joint_trajectory",
        needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
        parameters=(("sync_start_s", "s", 1.0), ("sync_end_s", "s", 1.3), ("tolerance_m", "m", 0.0001)),
        segments_join="ends_moving",
        exists_in=(f"{CONVEYOR}:72",),
        prior_art=(),
        priority="next",
        notes="Added 2026-11-18, promoted from lane F's test stand-in: no row ended moving at a target, so track_and_release had "
              "nothing in the catalog to follow. chain refuses a follower that does not start in its exact end state.",
    ),
    TrajectoryKind(
        id="track_and_release", name="Track, then leave a moving point", family="task_cycle",
        one_line="Already moving with a point, keep with it for a dwell in its frame, then leave it along a direction to rest: "
                 "the second half of a moving-target cycle.",
        use_cases=("tracking_eval", "conveyor_pick_and_place"),
        starts_as="path", dimensions=("1D", "2D", "3D", "6D"),
        space="cartesian", scope="cartesian_path", output="joint_trajectory",
        needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
        parameters=(("dwell_s", "s", 0.3), ("leave_m", "m", 0.05)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{CONVEYOR}:72",),
        prior_art=(),
        priority="later",
    ),
    TrajectoryKind(
        id="moving_pick_and_place", name="U between points that move", family="task_cycle",
        one_line="The U of pick_and_place_cycle (approach, down, dwell, up, over, down, dwell, up) where the first point moves: "
                 "down and up are tracked in the point's frame, synchronised at the bottom.",
        use_cases=("conveyor_pick_and_place", "cycle_time", "tracking_eval"),
        starts_as="path", dimensions=("3D", "6D"),
        space="cartesian", scope="coordinated_multi_joint", output="joint_trajectory",
        needs=(*LIMITS, "forward_kinematics", "inverse_kinematics"),
        parameters=(("lift_m", "m", 0.05), ("sync_window_s", "s", 0.5), ("tolerance_m", "m", 0.001)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{CONVEYOR}:72", f"{REACH_DATASET}:49"),
        prior_art=(),
        priority="next",
        notes="bam_grasp's conveyor line delivered 19/40 on a moving belt against 30/40 static: the loss is tracking error "
              "along the belt, which this kind's sync window makes a measured, gated number.",
    ),
    TrajectoryKind(
        id="adept_cycle", name="Adept cycle", family="task_cycle",
        one_line="The industry cycle-time benchmark: 25 mm up, 305 mm across, 25 mm down and back (extended: 25 / 700 / 25 mm).",
        use_cases=("cycle_time", "speed_limits"),
        # WHY path: a cycle-time benchmark: the timing is the RESULT (as fast as the limits allow), not the content
        starts_as="path", dimensions=("2D",),  # up, across, down: a vertical plane
        space="cartesian", scope="cartesian_path", output="joint_trajectory", needs=(*TIMED, "reach_map", "inverse_kinematics"),
        parameters=(("up_m", "m", 0.025), ("across_m", "m", 0.305), ("arrival_tolerance_m", "m", 0.00025)),
        segments_join="periodic_returns_to_start",
        exists_in=(),
        prior_art=(ADEPT, SCHNEIDER),
        priority="later",
        notes="The vendor-neutral number benchmark robots publish; bam_trajgen_bench's profiles (Glacier, AMP, Recycleye) are scored on SPPM.",
    ),
    TrajectoryKind(
        id="touch_down_contact", name="Touch-down on a surface", family="task_cycle",
        one_line="Descend from above a surface, the last stretch at a constant touch speed aiming slightly below it, dwell in contact, lift: "
                 "peak force, bounces and press-force error.",
        use_cases=("contact_eval", "pid_tuning", "tracking_eval", "conveyor_pick_and_place"),
        # WHY path: the touch speed is a speed you pick for the last stretch (the rungs 0.05 / 0.10 / 0.20 m/s): still a path
        starts_as="path", dimensions=("1D",),  # a vertical line down and back up
        space="cartesian", scope="cartesian_path", output="joint_trajectory", needs=("joint_limits", "velocity_limits", "reach_map", "inverse_kinematics"),
        parameters=(("approach_height_m", "m", 0.04), ("touch_speed_m_s", "m/s", 0.10), ("touch_stretch_m", "m", 0.02), ("aim_below_m", "m", 0.01),
                    ("dwell_s", "s", 0.8)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{RUNGS}:336",),
        prior_art=(),
        priority="later",
        notes="robot_controllers' R3 belt-impact rungs at 0.05 / 0.10 / 0.20 m/s on a belt moving 0.25 m/s.",
    ),
    # ============================================================ hold
    TrajectoryKind(
        id="hold_at_pose", name="Hold and read the settled stop", family="hold",
        one_line="Hold the last target for a set time and read the settled stop from the hold's final window: the segment that joins two "
                 "moves, and the stop-accuracy reading itself.",
        use_cases=("stop_accuracy", "pid_tuning", "repeatability"),
        starts_as="time_signal",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=("joint_limits",),
        parameters=(("hold_s", "s", 1.0), ("settle_window_s", "s", 0.3), ("approach_side", "text", "fixed")),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{FIVEBAR}/tapes.py:427", f"{ROBOT_MSGS}/control/joint_command.py:88"),
        prior_art=(ISO_9283, ISO_230_2),
        priority="now",
        notes="Zach's 1 mm tool gate at a settled stop (sim2real.result.TOOL_GATE_MM). Approaching every stop from one fixed side (the 11-15 "
              "READY aim: 1.48 -> 0.38 mm) is a parameter of the hold, as ISO 230-2's unidirectional approach is.",
    ),
    TrajectoryKind(
        id="disturbance_push", name="Hold under a push", family="hold",
        one_line="Hold a pose and push it (a set force for a set time in sim, by hand on the rig): recovery time and overshoot.",
        use_cases=("disturbance_rejection", "pid_tuning"),
        # WHY time_signal: a hold plus a push schedule in time
        starts_as="time_signal",
        space="joint", scope="coordinated_multi_joint", output="joint_trajectory", needs=("joint_limits", "home_pose"),
        parameters=(("push_n", "N", 15.0), ("push_s", "s", 0.4), ("push_at_s", "s", 0.5)),
        segments_join="starts_and_ends_at_rest",
        exists_in=(f"{RUNGS}:495", f"{FIVEBAR}/gravity.py:161"),
        prior_art=(MOTEUS_PID,),
        priority="later",
        notes="The trajectory is a plain hold; the generated part is the disturbance schedule beside it (a sim force, or a cue to push).",
    ),
    TrajectoryKind(
        id="weightless_float", name="Weightless float", family="hold",
        one_line="Hold a pose under PD with gravity feed-forward, then drop kp and kd to zero and keep sending the gravity torque: the drift "
                 "over the float window is how well gravity is modelled.",
        use_cases=("gravity_id", "smoke_test"),
        starts_as="time_signal",
        space="torque", scope="coordinated_multi_joint", output="command_sequence", needs=("torque_limits", "forward_kinematics", "dynamics_model"),
        parameters=(("float_s", "s", 5.0), ("poses", "count", 5)),
        segments_join="ends_anywhere",
        exists_in=(f"{FIVEBAR}/gravity.py:91",),
        prior_art=(MA_HOLLERBACH,),
        priority="next",
        notes="'It feels weightless' made measurable (deg/s of drift); hardware-proven on the gravity-comp package (-0.06 deg drift).",
    ),
)

KINDS: dict[str, TrajectoryKind] = {kind.id: kind for kind in CATALOG}
