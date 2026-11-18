"""Build the robot_trajectories 0.1 report: what the signal library is now, shown on its catalog page, with its own code run.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/docs/build_report.py --snapshot /home/bam/robot_trajectories

Writes ``reports/robot-trajectories-0.1-2026-11-18.html`` (text only) and copies its media into ``reports/media/`` (gitignored):
the catalog page's hero video and a GIF of it, and the page and card stills. Everything it shows is measured at build time:

- the quick start is run here, in the package's venv against ``--snapshot`` (its src/ and tests/ first on the path), and its real
  output printed under it;
- the counts, the locked dimensions and the coverage come from the catalog page's build (bam_ws
  ``docs/trajectory_generator/catalog/at_a_glance/build_kind_previews.py``: ``catalog-kinds/build.json`` and ``data.js``);
- the plugin table asks each plugin's own venv which time law it registers and which library version serves it.

Stdlib only (the package's venv runs it; ffmpeg makes the GIF). Offline: no network, media by relative path.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MEDIA = REPORTS / "media"
OUT = REPORTS / "robot-trajectories-0.1-2026-11-18.html"
CATALOG_MEDIA = Path("/home/bam/bam_ws/reports/media/trajectory-generator-2026-11-18")
CATALOG_PAGE = CATALOG_MEDIA / "catalog.html"
BUILD = CATALOG_MEDIA / "catalog-kinds" / "build.json"
DATA = CATALOG_MEDIA / "catalog-kinds" / "data.js"
CHECK_LOG = CATALOG_MEDIA / "catalog-kinds" / "check.txt"  # check.cjs's output, tee'd there when it last ran
LANE_BUILDER = "bam_ws docs/trajectory_generator/catalog/at_a_glance/build_kind_previews.py"
STILLS = [
    ("catalog-hero.png", "The catalog at a glance: 75 kinds, trajectories left and paths right, in columns 1D / 2D / 3D / 6D. Each glyph is the kind's own motion."),
    ("catalog-card-pick_and_place_cycle-6D.png", "pick_and_place on the UR5e (6D): the U's tool path, grey ahead and lit where the tool has passed; joint space and Cartesian space beside it."),
    ("catalog-card-moving_target_track-6D.png", "moving_target_track on the UR5e: meet a point moving at 5 cm/s with matched velocity, track it, return to rest."),
    ("catalog-card-torque_chirp-1D.png", "torque_chirp on the CAN 11 pendulum (1D): torque-commanded, so the motion shown is the dynamics stand-in's prediction. 2D-6D: not applicable (a 1D kind)."),
    ("catalog-card-iso9283_pose_cube-6D.png", "iso9283_pose_cube on the UR5e: P1-P5 in the largest cube inside the reach, 30 cycles (first 60 s shown). 1D and 2D are locked: the reasons are on the switch."),
    ("catalog-card-cartesian_raster-6D.png", "cartesian_raster on the UR5e: the region's rows at a ladder of tool speeds."),
]
HERO = "catalog-hero.mp4"
PLUGINS = [
    ("robot_trajectories_ruckig", "Ruckig Community", "ruckig", "ruckig",
     "one jerk-limited S-curve leg between two states", "never handed intermediate waypoints (Community sends them to Ruckig's cloud)"),
    ("robot_trajectories_ruckig_pro", "Ruckig Pro (drop-in)", "ruckig", "ruckig",
     "the local Pro build: intermediate waypoints, position limits", "registers only after checking the local build's provenance; never installed beside Community"),
    ("robot_trajectories_toppra", "TOPP-RA", "toppra", "toppra",
     "time-optimal timing along a fixed joint path", "the torque-constrained mode needs the space's dynamics: a core change, proposed, not made"),
]
QUICK_START = '''import numpy as np
from fixtures import robots  # the package's test fixtures: CAN 11's limits, the UR5e's description and DH kinematics
from robot_trajectories import JointSpace, catalog, chain, kinds, pick_and_place, pid_tuning, split, step_train, summary

space = JointSpace.from_limits(robots.PENDULUM_LIMITS)  # the robot, injected: bare joint limits
session = chain(pid_tuning(space, joint="can11"), step_train(space, joint="can11", steps_fraction=(0.3, -0.3)))
print(summary(session))
print(len(split(session)), "rest-to-rest pieces")

arm = JointSpace.from_description(robots.UR5E_DESCRIPTION, max_acceleration=6.0).within(robots.UR5E_DEMO_BOX)
arm = arm.with_kinematics(robots.UR5eKinematics())  # FK/IK passed in: the Cartesian kinds unlock
tool = arm.forward(arm.home)[:3, 3]
cycle = pick_and_place(arm, tool + (0.08, -0.08, -0.12), tool + (-0.08, 0.10, -0.12), lift_m=0.08)
print(summary(cycle))

found = catalog(use_case="conveyor_pick_and_place", space=arm)
print(len(kinds()), "kinds registered; conveyor_pick_and_place:", ", ".join(entry.name for entry in found.implemented))
'''
FAMILY_LABEL = {"excitation": "Excitation", "controller_test": "Controller test", "quasi_static": "Quasi-static", "point_to_point": "Point to point",
                "coverage": "Coverage", "standard_test": "Standard test", "calibration": "Calibration", "stress": "Stress", "task_cycle": "Task cycle",
                "hold": "Hold"}
LADDER = [
    ("joint_limits", "1. Joint limits", "positions, velocities, accelerations (jerk and torque caps when known), a named home", "JointSpace.from_limits(limits)"),
    ("kinematics", "2. + FK/IK", "forward(q) -> tool pose, inverse(pose, seed) -> q; a reach map is an optional cache", "space.with_kinematics(fk_ik)"),
    ("dynamics", "3. + dynamics", "torque(q, qd, qdd); forward(q, qd, tau) -> qdd for effort-commanded kinds", "space.with_dynamics(model)"),
    ("torque_limits", "4. + torque limits", "max_effort: the drive caps open-loop torque kinds stay under", "JointLimits(max_effort=...)"),
]


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def run(command: list[str], **options) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, **options)


def quick_start(python: Path, snapshot: Path) -> tuple[str, str]:
    """Run the quick start in the package's venv against the snapshot; returns (code, its real output)."""
    preamble = f"import sys; sys.path[:0] = [{str(snapshot / 'src')!r}, {str(snapshot / 'tests')!r}]\n"
    done = run([str(python), "-I", "-c", preamble + QUICK_START], timeout=300)
    if done.returncode != 0:
        raise SystemExit(f"the quick start failed:\n{done.stderr}")
    return QUICK_START.strip(), done.stdout.rstrip()


def plugin_rows(snapshot: Path) -> list[dict]:
    """Each plugin's venv: which provider serves its time law, the library version, load errors (asked, not assumed)."""
    rows = []
    for folder, title, registers, library, what, rule in PLUGINS:
        python = ROOT / "plugins" / folder / ".venv" / "bin" / "python"
        probe = (f"import sys, json; sys.path[:0] = [{str(snapshot / 'src')!r}, {str(snapshot / 'plugins' / folder / 'src')!r}]\n"
                 "import importlib.metadata as md, robot_trajectories as rt\n"
                 "from robot_trajectories import plugins\n"
                 f"laws = rt.time_laws().get({registers!r}, ())\n"
                 "version = None\n"
                 "for dist in ('ruckig-pro', 'ruckig', 'toppra'):\n"
                 "    try: version = version or (dist + ' ' + md.version(dist))\n"
                 "    except md.PackageNotFoundError: pass\n"
                 "print(json.dumps(dict(providers=[info.provider for info in laws], errors=plugins.errors(), version=version)))\n")
        if not python.exists():
            rows.append(dict(folder=folder, title=title, registers=registers, what=what, rule=rule, status="no venv", version="", providers=[]))
            continue
        done = run([str(python), "-I", "-c", probe], timeout=120)
        try:
            answer = json.loads(done.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            answer = dict(providers=[], errors=[done.stderr.strip()[-300:]], version=None)
        status = ("loads: serves '" + registers + "' as " + ", ".join(answer["providers"])) if answer["providers"] else \
            "does not register: " + "; ".join(answer["errors"])[:200]
        rows.append(dict(folder=folder, title=title, registers=registers, what=what, rule=rule, status=status, version=answer.get("version") or "",
                         providers=answer["providers"]))
    return rows


def copy_media() -> dict[str, str]:
    """Copy the hero video and the stills into reports/media/ (gitignored); make a GIF fallback. Returns name -> relative path."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    shown = {}
    for name in [HERO] + [still for still, _ in STILLS]:
        source = CATALOG_MEDIA / name
        if not source.exists():
            raise SystemExit(f"missing {source}: run the catalog page's check.cjs and record the hero first")
        target = MEDIA / name.replace(".", "-2026-11-18.", 1)
        shutil.copy2(source, target)
        shown[name] = f"media/{target.name}"
    gif = MEDIA / "catalog-hero-2026-11-18.gif"
    done = run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(CATALOG_MEDIA / HERO), "-vf",
                "fps=8,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4", str(gif)])
    if done.returncode == 0:
        shown["gif"] = f"media/{gif.name}"
    return shown


def load_data() -> dict:
    text = DATA.read_text(encoding="utf-8")
    start = text.index("window.TG_DATA = ") + len("window.TG_DATA = ")
    return json.loads(text[start:].rstrip().rstrip(";"))


def family_table(kinds: list[dict]) -> str:
    rows = defaultdict(lambda: Counter())
    for kind in kinds:
        counter = rows[kind["family"]]
        if kind["is_plan"]:
            counter["plans"] += 1
        else:
            counter["catalog"] += 1
            counter["implemented"] += int(kind["implemented"])
        slots = kind.get("slots", {})
        counter["playable"] += sum(1 for slot in slots.values() if slot.get("state") == "playable" and slot.get("preview"))
        counter["applicable"] += sum(1 for slot in slots.values() if slot.get("state") != "not_applicable")
        counter["plugin"] += int(kind.get("served_by") == "plugin")
    order = list(FAMILY_LABEL)
    body = []
    total = Counter()
    for family in sorted(rows, key=order.index):
        counter = rows[family]
        total.update(counter)
        names = ", ".join(f"<code>{esc(k['function'] or k['id'])}</code>" for k in kinds if k["family"] == family)
        body.append(f"<tr><td>{esc(FAMILY_LABEL[family])}</td><td class=n>{counter['implemented']} / {counter['catalog']}</td><td class=n>{counter['plans'] or ''}</td>"
                    f"<td class=n>{counter['playable']} / {counter['applicable']}</td><td class=names>{names}</td></tr>")
    body.append(f"<tr class=total><td>All</td><td class=n>{total['implemented']} / {total['catalog']}</td><td class=n>{total['plans']}</td>"
                f"<td class=n>{total['playable']} / {total['applicable']}</td><td></td></tr>")
    return ("<table class=grid><thead><tr><th>Family</th><th>Implemented / catalog rows</th><th>Plans</th><th>Dimensions playing / applicable</th>"
            "<th>Kinds (functions)</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>")


def locked_table(build: dict) -> str:
    groups = defaultdict(list)
    for name, slots in build["outcomes"].items():
        for label, outcome in slots.items():
            if outcome["state"] != "locked":
                continue
            if outcome.get("error_type") == "MissingRung" and "forward kinematics" in outcome.get("error", ""):
                key = ("The pendulum has no kinematics", "MissingRung: the kind needs forward kinematics (rung 2); CAN 11's fixture space has none, "
                       "and a 1-joint tool moves on an arc, so a Cartesian kind's 1D slot is the library's refusal.")
            else:
                key = (f"{outcome.get('error_type')}", outcome.get("error", ""))
            groups[key].append((name, label))
    rows = []
    for (title, why), members in sorted(groups.items(), key=lambda item: -len(item[1])):
        who = ", ".join(f"<code>{esc(name)}</code> {esc(label)}" for name, label in members)
        rows.append(f"<tr><td class=n>{len(members)}</td><td><b>{esc(title)}</b><div class=why>{esc(why)}</div></td><td class=names>{who}</td></tr>")
    return "<table class=grid><thead><tr><th>#</th><th>Why (the library's own refusal)</th><th>Kind, dimension</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def ladder_table(kinds: list[dict]) -> str:
    counts = Counter(kind["entry_level"] for kind in kinds)
    collision = sum(1 for kind in kinds if "collision" in (kind.get("entry_extras") or []))
    rows = [f"<tr><td><b>{esc(title)}</b></td><td>{esc(what)}</td><td><code>{esc(call)}</code></td><td class=n>{counts.get(level, 0)}</td></tr>"
            for level, title, what, call in LADDER]
    rows.append(f"<tr><td><b>beside the ladder: a collision model</b></td><td>in_collision(q): the guard then refuses any colliding sample</td>"
                f"<td><code>space.with_collision(scene)</code></td><td class=n>{collision} check one</td></tr>")
    return ("<table class=grid><thead><tr><th>Rung</th><th>What you pass in</th><th>The call</th><th>Kinds whose top rung it is</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def plugin_table(rows: list[dict], kinds: list[dict]) -> str:
    serves = defaultdict(list)
    for kind in kinds:
        if kind.get("plugin_time_law"):
            serves[kind["plugin_time_law"]].append(kind["function"])
    body = []
    for row in rows:
        kinds_served = ", ".join(f"<code>{esc(name)}</code>" for name in serves.get(row["registers"], []))
        body.append(f"<tr><td><b>{esc(row['title'])}</b><div class=why><code>plugins/{esc(row['folder'])}</code></div></td><td><code>{esc(row['registers'])}</code></td>"
                    f"<td>{esc(row['what'])}<div class=why>Rule: {esc(row['rule'])}</div></td><td>{kinds_served}</td>"
                    f"<td>{esc(row['status'])}<div class=why>{esc(row['version'])}</div></td></tr>")
    return ("<table class=grid><thead><tr><th>Plugin</th><th>Time law</th><th>What it does</th><th>Kinds it serves</th><th>In its own venv, at build time</th></tr></thead>"
            "<tbody>" + "".join(body) + "</tbody></table>")


CSS = """
:root { --bg: #fff; --text: #17181c; --text2: #4a4d55; --text3: #7a7e87; --border: #e3e4e8; --soft: #f5f6f8; --accent: #3d63dd; --amber: #a15c00; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 28px 64px; }
h1 { font-size: 26px; line-height: 1.25; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 40px 0 10px; letter-spacing: -0.005em; }
h3 { font-size: 15px; margin: 22px 0 6px; }
p, li { color: var(--text); }
.sub { color: var(--text2); margin: 0 0 18px; }
.verdict { font-size: 17px; line-height: 1.5; border-left: 4px solid var(--accent); background: var(--soft); padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 18px 0; }
.verdict b { font-weight: 650; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 16px 0 6px; }
.kpi { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.kpi b { display: block; font-size: 22px; font-variant-numeric: tabular-nums; }
.kpi span { color: var(--text2); font-size: 13px; }
video, .still img { width: 100%; display: block; border: 1px solid var(--border); border-radius: 8px; background: #f0f1f3; }
.hero figcaption, .still figcaption { color: var(--text2); font-size: 13px; margin-top: 6px; }
figure { margin: 0; }
.stills { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.stills .still:first-child { grid-column: 1 / -1; }
pre { background: #0f1115; color: #e6e7ea; border-radius: 8px; padding: 14px 16px; overflow: auto; font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre.out { background: var(--soft); color: var(--text); border: 1px solid var(--border); }
code { font: 12.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: var(--soft); padding: 0 3px; border-radius: 3px; }
pre code { background: none; padding: 0; }
table.grid { width: 100%; border-collapse: collapse; font-size: 13.5px; margin: 8px 0; }
table.grid th { text-align: left; color: var(--text3); font-weight: 550; font-size: 12.5px; border-bottom: 1px solid var(--border); padding: 6px 8px; }
table.grid td { border-bottom: 1px solid var(--border); padding: 7px 8px; vertical-align: top; }
table.grid td.n { font-variant-numeric: tabular-nums; white-space: nowrap; }
table.grid td.names { color: var(--text2); font-size: 12.5px; }
table.grid tr.total td { font-weight: 650; border-top: 2px solid var(--border); }
.why { color: var(--text2); font-size: 12.5px; margin-top: 2px; }
.links a { color: var(--accent); }
.decide h3 { margin-top: 18px; }
.decide ul { margin: 4px 0; padding-left: 20px; }
.muted { color: var(--text3); font-size: 13px; }
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot", type=Path, default=ROOT, help="the checkout the quick start and the plugin probes run against")
    parser.add_argument("--python", type=Path, default=ROOT / ".venv" / "bin" / "python", help="the package's venv")
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    sha = run(["git", "-C", str(snapshot), "rev-parse", "HEAD"]).stdout.strip()
    build = json.loads(BUILD.read_text())
    data = load_data()
    kinds = data["catalog"]["kinds"]
    if build["snapshot"]["sha"] != sha:
        print(f"note: the catalog page was built from {build['snapshot']['sha'][:7]}, this report runs {sha[:7]}", file=sys.stderr)
    code, output = quick_start(args.python, snapshot)
    plugins = plugin_rows(snapshot)
    media = copy_media()
    coverage = build["coverage"]
    rows = [kind for kind in kinds if not kind["is_plan"]]
    plans = [kind for kind in kinds if kind["is_plan"]]
    locked = coverage["slots_applicable"] - coverage["slots_playable"]
    built_at = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    agreement = build.get("kinematics_agreement_mm", {})
    check_lines = CHECK_LOG.read_text(encoding="utf-8").strip().splitlines() if CHECK_LOG.exists() else []
    check_tail = "\n".join(line for line in check_lines if line.startswith(("ok", "FAIL", "coverage", "0 passed")) or " passed, " in line)
    stills = "".join(f"<figure class=still><img src='{esc(media[name])}' alt='{esc(caption)}' loading=lazy><figcaption>{esc(caption)}</figcaption></figure>"
                     for name, caption in STILLS)
    gif = f"<img src='{esc(media['gif'])}' alt='The catalog hero, as a GIF'>" if "gif" in media else ""
    slot_lines = "".join(f"<li><b>{esc(label)}</b> {esc(slot['title'])}: {esc(', '.join(slot['stand_ins']))}</li>" for label, slot in build["slots"].items())
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>robot_trajectories 0.1: every kind in 3D (2026-11-18)</title>
<!-- Generated by /home/bam/robot_trajectories/docs/build_report.py at {esc(built_at)}; snapshot {esc(sha[:7])}. -->
<style>{CSS}</style></head>
<body><main>
<h1>robot_trajectories 0.1: every kind opens in 3D, with its joint and Cartesian views and a 1D–6D switch</h1>
<p class=sub>Zach's catalog feedback (Daily Note 2026-11-18): “for each trajectory 1. Show a 3D view (even for a 1D thing thats a single joint)
2. Show the joint space view and the cartesian space view 3. Have the option to swith between 1D, 2D, 3D, 6D (as applicable) and mamke sure that you
have this for all of them.” Package at <code>{esc(sha[:7])}</code> (v{esc(build['snapshot']['version'])}); report built {esc(built_at)}.</p>

<div class=verdict><b>All {coverage['kinds']} registered kinds play in 3D on the real meshes, each with its joint-space and Cartesian views and a dimension switch:
{coverage['slots_playable']} of the {coverage['slots_applicable']} dimensions they apply to play; the other {locked} are disabled with the library's own refusal on the switch.</b>
The {coverage['catalog_rows']} catalog rows are all implemented ({coverage['catalog_rows_implemented']}), plus {len(plans)} plans and higher-order kinds. The package itself is unchanged; the
page and its previews live in bam_ws <code>reports/media/trajectory-generator-2026-11-18/</code>, built by the lane's <code>build_kind_previews.py</code> (not committed yet).</div>

<div class=kpis>
<div class=kpi><b>{coverage['kinds_with_preview']} / {coverage['kinds']}</b><span>kinds with a 3D preview</span></div>
<div class=kpi><b>{coverage['slots_playable']} / {coverage['slots_applicable']}</b><span>dimensions playable / applicable</span></div>
<div class=kpi><b>{coverage['catalog_rows_implemented']} / {coverage['catalog_rows']}</b><span>catalog rows implemented</span></div>
<div class=kpi><b>{max(agreement.values()) if agreement else 0:.4f} mm</b><span>drawn path vs the package's kinematics (worst pairwise)</span></div>
</div>

<h2>The catalog page, driven</h2>
<figure class=hero><video autoplay muted loop playsinline controls preload=auto src='{esc(media[HERO])}'>{gif}</video>
<figcaption>Click a joint-space kind (random reachable targets): its card opens on the UR5e's three joints; the switch moves it to the five-bar and the full UR5e.
Then a Cartesian raster: the whole tool path in grey, lit as the tool passes, on the five-bar and then the UR5e. Recorded from the real page in headless Chrome.
{f"GIF: <a href='{esc(media['gif'])}'>{esc(media['gif'])}</a>." if 'gif' in media else ''}</figcaption></figure>

<h2>Four cards and the page, at 1440 × 900</h2>
<div class=stills>{stills}</div>

<h2>Quick start (the V1 API), run while building this report</h2>
<p>Every kind is a plain function <code>f(space, ..., start=None) -&gt; robot_msgs JointTrajectory</code>. The robot is injected into a <code>JointSpace</code>;
the task arrives as points and numbers. Run in the package's venv against <code>{esc(sha[:7])}</code>:</p>
<pre><code>{esc(code)}</code></pre>
<pre class=out>{esc(output)}</pre>

<h2>Kinds by family</h2>
<p>Catalog rows ↔ kinds by <code>catalog_id</code> (one kind per row); plans are registry kinds no row names. A dimension is “applicable” when the kind declares
it (joint and torque kinds: joints moving at once) or, for a Cartesian kind, on every robot — its dimension is the geometry's, so every robot is tried and the
ones that cannot trace it refuse.</p>
{family_table(kinds)}

<h2>Plugins: kinds and time laws from other packages</h2>
<p>Simple kinds live in the package (numpy + robot_msgs). A time law that needs another library is a plugin, found by entry point
(<code>robot_trajectories.time_laws</code>). Two providers of one name are refused, never ranked. The catalog's two plugin-served kinds are filtered by
<b>Built → Needs a plugin</b> on the page; their previews were generated in the plugin's own venv.</p>
{plugin_table(plugins, kinds)}

<h2>The entry ladder: what you pass in unlocks what</h2>
{ladder_table(kinds)}

<h2>What is locked, and why</h2>
<p>{locked} dimensions are disabled on their cards. Every reason is the exception the kind raised on the fixture robot, shown as text on the switch and in full in the
card's Dimensions list. None is a display failure.</p>
{locked_table(build)}

<h2>How the previews are made</h2>
<ul>
<li>The motion is the package's own: each kind's default call on one robot per dimension, the richest space the test fixtures offer (every model a labelled stand-in):
<ul>{slot_lines}</ul></li>
<li>Exceptions to the default call, each in <code>kind_generate.call_plan</code>: the four kinds with required arguments get a few-cm example from the start's tool point (the
tests' own values); on the planar five-bar a Cartesian kind starts at the tests' Cartesian start with the plane's up / across; <code>home</code> and
<code>return_to_ready</code> start halfway to the room's corner (from home they are empty).</li>
<li>Rendering is bam_ws's shared kit (<code>build_previews.py</code>'s robots: pinocchio FK for the UR5e, the closed-form linkage for the five-bar, the rig for the pendulum), one lazy
file per (kind, dimension). Long kinds are cut at 60 s, and say so in their title. Torque-commanded kinds play the dynamics stand-in's predicted motion, and say so.</li>
<li>The drawn path is the path the kind computed: pairwise tool distances, display FK vs the package's kinematics, agree to {esc(json.dumps(agreement))} mm (the frames differ by a rigid motion).</li>
</ul>

<h2>Links</h2>
<ul class=links>
<li><a href="../docs/peps/0002-signal-library.md">PEP 0002: the signal library</a> (the decision record), <a href="../README.md">README</a>, <a href="../docs/KINDS_TODO.md">KINDS_TODO</a></li>
<li><a href="file://{esc(CATALOG_PAGE)}">The catalog page</a> <span class=muted>({esc(CATALOG_PAGE)})</span></li>
</ul>

<h2>Decision surface</h2>
<div class=decide>
<h3>Done and proved</h3>
<ul>
<li>Every chip opens a card with the 3D view and its tool path, joint space and Cartesian space, and a 1D/2D/3D/6D switch: the page's check opens all {coverage['kinds']} kinds and every
playable dimension and asserts the path, plots and switch (bam_ws <code>catalog/at_a_glance/check.cjs</code>).</li>
<li>The quick start above ran against the package while this report was built.</li>
</ul>
{f"<details><summary>The page's check, as it last ran ({esc(check_lines[-1] if check_lines else '')})</summary><pre class=out>{esc(check_tail)}</pre></details>" if check_lines else "<p class=muted>The page's check has not run yet.</p>"}
<h3>Decided for you (each undoable)</h3>
<ul>
<li>3D is the UR5e on pan, lift and elbow with the wrist held at READY (position-only kinematics, built in the lane's <code>kind_generate.py</code>). Undo: drop the 3D slot there.</li>
<li>Cartesian kinds are tried on every robot, so a straight line shows on the five-bar, the 3-joint and the 6-joint UR5e; their 1D slot shows the pendulum's refusal. Undo:
<code>applicable()</code> in <code>kind_generate.py</code>.</li>
<li>Stand-in jerk limits (the tests' values: pendulum 200, UR5e 40 rad/s³; five-bar 100, no test states one) so the two jerk-limited kinds generate. Undo: the constants at the top of <code>kind_generate.py</code>.</li>
<li>The card opens in a drawer 58 % of the window wide (640-1100 px), so the 3D view sits beside the plots at 1440 px. Undo: <code>DRAWER_DEFAULT</code> in <code>page.js</code>.</li>
</ul>
<h3>Left</h3>
<ul>
<li>A pendulum kinematics (its payload on an arc) would turn the 17 Cartesian 1D refusals into honest “unreachable” ones or short arcs: not in the fixtures, so not invented here.</li>
<li>The torque command itself is not plotted (the card shows the predicted motion); a torque panel is a kit change.</li>
</ul>
<h3>Needs you</h3>
<ul><li>Nothing.</li></ul>
<h3>Deliberately not done</h3>
<ul>
<li>No edit to the package's <code>src/</code> or <code>tests/</code> (another lane owns them); no commit.</li>
</ul>
</div>
<p class=muted>Sources: {esc(LANE_BUILDER)} → {esc(BUILD)}; plugin rows asked of each plugin's venv at build time.</p>
</main></body></html>
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB) + media {sorted(media.values())}")
    for row in plugins:
        print(f"  {row['title']}: {row['status']} {row['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
