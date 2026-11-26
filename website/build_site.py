"""Build the landing page (GitHub Pages) from real runs: the API at a glance, every kind with its joint positions, the names.

    uv run --isolated --no-project --with-editable /home/bam/robot_msgs --with-editable /home/bam/robot_trajectories --with-editable /home/bam/robot_trajectories/plugins/robot_trajectories_ruckig --with-editable /home/bam/robot_trajectories/plugins/robot_trajectories_toppra --with matplotlib --with pygments python /home/bam/robot_trajectories/website/build_site.py && python3 /home/bam/robot_trajectories/website/publish_pages.py

The first half writes website/_build/ (index.html, img/); the second commits it to gh-pages and pushes it. Outside Zach's
machine, swap the two robot_msgs / robot_trajectories editables for ``--with robot-msgs`` and a clone's own paths.

Every code block on the page is code this script ran (``session.py``'s Session, ported from robot_metrics), and every
picture is a plot of the trajectory that call returned. The glance is ``examples/quick_start.py``, run as is. Each catalog
card is one call of one kind on the first of four spaces it runs on (the CAN 11 pendulum's bare limits from
``examples/can11_limits.json``; the same pendulum with the stand-in drive limit, home and dynamics of
``tests/fixtures/lane_b.py``; a UR5e's joint limits; the UR5e with the stand-in kinematics and gravity of
``tests/fixtures``). A kind that runs on none gets a card with the refusal it raised, never a faked plot. matplotlib,
pygments and the Ruckig / TOPP-RA plugins are build-time only: the package's runtime stays numpy + robot_msgs (the leaf rule).

WHY the highlighting and the notebook rhythm (Zach, /rate bad 2026-11-18, on robot_metrics' page): Pygments at build
time, one statement then its own output in a quieter band, blank lines between ideas, the API at a glance first.
WHY degrees on every plot: "SI in code; degrees on every display" (CLAUDE.md).
"""

from __future__ import annotations

import ast
import html
import inspect
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = Path(os.environ.get("ROBOT_TRAJECTORIES_SITE_OUT") or HERE / "_build")
IMG = OUT / "img"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tests"))  # tests/fixtures: the round's stand-in robots, read by the cards' spaces

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402 - matplotlib's own dependency
from pygments import highlight  # noqa: E402
from pygments.formatters import HtmlFormatter  # noqa: E402
from pygments.lexers import PythonLexer, TOMLLexer  # noqa: E402
from session import Session  # noqa: E402

import robot_trajectories  # noqa: E402
from robot_trajectories.trajectory_catalog import kind as vocabulary  # noqa: E402
from robot_trajectories.trajectory_catalog.catalog import USE_CASES  # noqa: E402

REPO = "https://github.com/zacharyyamaoka/robot_trajectories"
SITE = "https://zacharyyamaoka.github.io/robot_trajectories/"
PYPI = "https://pypi.org/project/robot-trajectories/"
PEP = f"{REPO}/blob/main/docs/peps/0002-signal-library.md"
VERSION = robot_trajectories.__version__
PRIVATE = re.compile(r"/home/|/archive|/Users/|@gmail|\.netrc|wandb_api|gh[pousr]_[A-Za-z0-9]{20}")

# WHY the reference palette's categorical slots in fixed order (the dataviz skill's validated light set): the plots sit on
# white paper in both themes, so one light set; a joint keeps its colour on every card.
JOINT_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
PLOT_POINTS = 1500
PLOT_DPI = 120


# ---- rendering helpers (robot_metrics' website/build_site.py) -----------------------------------------------------

LEXER, FORMATTER = PythonLexer(), HtmlFormatter(nowrap=True)
WRAP_OVER = 96


def python(code: str) -> str:
    return highlight(code, LEXER, FORMATTER).rstrip("\n")


def output_block(text: str) -> str:
    wrap = "\n" not in text and len(text) > WRAP_OVER
    return f"<pre class='out{' wrap' if wrap else ''}'>{html.escape(text)}</pre>"


def notebook(cell: dict, script: str) -> str:
    """A cell as a notebook: statements highlighted, each output in a quieter band right under its statement.
    Statements with no output share a block, with the source's blank lines between ideas."""

    blocks, group, group_gap = [], [], False
    for statement in cell["statements"]:
        if not group:
            group_gap = statement["gap_before"] and bool(blocks)
        elif statement["gap_before"]:
            group.append("")
        group.append(statement["code"])
        if statement["output"]:
            blocks.append(f"<pre class='src{' gap' if group_gap else ''}'>{python(chr(10).join(group))}</pre>")
            blocks.append(output_block(statement["output"]))
            group = []
    if group:
        blocks.append(f"<pre class='src{' gap' if group_gap else ''}'>{python(chr(10).join(group))}</pre>")
    return f"<div class='notebook' data-script='{script}'>{''.join(blocks)}</div>"


def token_css() -> str:
    light = HtmlFormatter(style="default").get_token_style_defs(".src")
    dark_auto = HtmlFormatter(style="github-dark").get_token_style_defs(':root:not([data-theme="light"]) .src')
    dark_forced = HtmlFormatter(style="github-dark").get_token_style_defs(':root[data-theme="dark"] .src')
    return "\n".join([*light, "@media (prefers-color-scheme: dark) {", *dark_auto, "}", *dark_forced])


def esc(value: object) -> str:
    return html.escape(str(value))


def prose(text: str) -> str:
    """A docstring line as HTML: escaped, its ``literals`` as code."""
    return re.sub(r"``(.+?)``", r"<code>\1</code>", esc(text))


def words(tag: str) -> str:
    return tag.replace("_", " ")


def first_line(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    return " ".join(doc.split("\n\n")[0].split())


# ---- the scripts the page shows ------------------------------------------------------------------------------------

def without_docstring(code: str) -> str:
    """A script minus its module docstring (the page says what it is; the REPL echo of a string would be noise)."""
    tree = ast.parse(code)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        code = "\n".join(code.splitlines()[tree.body[0].end_lineno:])
    return code.strip("\n")


def run_script(script: str, cells: list[tuple[str, str, str]]) -> Session:
    """One script the page shows as one piece, run in a fresh namespace from the repo's root (a clone's layout)."""

    session = Session({"__name__": "__main__"})
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        for label, caption, code in cells:
            started = time.perf_counter()
            session.cell(label, caption, code)
            print(f"  {script} · {label}: {time.perf_counter() - started:.1f} s", flush=True)
    finally:
        os.chdir(previous)
    session.script = script
    return session


SPACES_CODE = '''
import json

import numpy as np

from fixtures import lane_b, robots          # tests/fixtures: the round's stand-in robots, never part of the package
from robot_trajectories import JointSpace
from robot_trajectories.space import limits_from_json

# The CAN 11 pendulum: bare joint limits, from examples/can11_limits.json
limits = limits_from_json(json.loads(open("examples/can11_limits.json").read())["limits"])
pendulum = JointSpace.from_limits(limits)

# The same pendulum plus a stand-in drive limit, a named home (the gravity bottom) and stand-in dynamics
pendulum_plus = JointSpace.from_limits(lane_b.pendulum_limits(), home=[0.0]).with_dynamics(lane_b.Can11Dynamics())

# A UR5e in a box above the table: joint limits only (its URDF has no acceleration, so one is stated)
ur5e = JointSpace.from_description(robots.UR5E_DESCRIPTION, max_acceleration=6.0).within(robots.UR5E_DEMO_BOX)

# The UR5e with every rung: stand-in DH kinematics and gravity, a drive and a jerk limit, a named home
box = robots.UR5E_DEMO_BOX.replace(max_effort=np.full(6, lane_b.UR5E_DRIVE_LIMIT_NM), max_jerk=np.full(6, 40.0))
ur5e_plus = (JointSpace.from_limits(box, home=0.5 * (box.min_position + box.max_position))
             .with_kinematics(robots.UR5eKinematics())
             .with_dynamics(lane_b.UR5E_DYNAMICS))
tool = ur5e_plus.forward(ur5e_plus.home)[:3, 3]      # the tool point at home, in metres

print(pendulum)
print(ur5e_plus)
'''

NAMES_CODE = '''
import json

import numpy as np

from robot_trajectories import TrajectoryGenerator, WaypointTrajectoryGenerator, chirp, waypoint_trajectory_generators
from robot_trajectories.space import limits_from_json

limits = limits_from_json(json.loads(open("examples/can11_limits.json").read())["limits"])

isinstance(chirp, TrajectoryGenerator)            # every kind is a TrajectoryGenerator: call it, get a Trajectory

generator = WaypointTrajectoryGenerator.minimum_jerk(peak_speed_fraction=0.6)
leg = generator(np.radians([[0.0], [15.0]]), limits, 500.0)    # points (M, J) + joint limits in, a JointTrajectory out
print(f"{type(leg).__name__}: {leg.t[-1]:.2f} s, {len(leg.t)} samples, peak {np.degrees(np.abs(leg.velocities).max()):.0f}°/s")
isinstance(generator, TrajectoryGenerator)        # so a WaypointTrajectoryGenerator is one too

sorted(waypoint_trajectory_generators())          # built in, plus whatever plugins are installed
'''

# The arguments a kind cannot default (a task's points); everything else takes the kind's own defaults.
EXTRA = {
    "cartesian_path": "[tool, tool + (0.0, 0.1, 0.0)]",
    "cartesian_waypoints": "[tool + (0.05, 0.0, 0.0), tool + (0.0, 0.08, -0.05)]",
    "pick_and_place": "pick=tool + (0.08, -0.08, -0.12), place=tool + (-0.08, 0.10, -0.12), lift_m=0.08",
    "joint_waypoints": "np.radians([[30, -70, 100, -80, -80, 30], [-30, -110, 60, -120, -100, -30]])",
}
SPACE_ABOUT = {
    "pendulum": "CAN 11 pendulum · bare limits",
    "pendulum_plus": "CAN 11 pendulum · + dynamics, drive limit, home",
    "ur5e": "UR5e · joint limits",
    "ur5e_plus": "UR5e · + kinematics, gravity, drive and jerk limits",
}
STAND_IN = {"pendulum_plus", "ur5e_plus"}


def spaces_for(info) -> tuple[str, ...]:
    """The spaces to try, simplest first: a kind with a 1-D form starts on the pendulum, a multi-joint one on the UR5e."""
    if "1D" in info.dimensions:
        return ("pendulum", "pendulum_plus", "ur5e_plus")
    return ("ur5e", "ur5e_plus")


def run_kind(name: str, info, namespace: dict) -> dict:
    """Call one kind on the first space it runs on; keep the call, the trajectory, or the refusals."""

    refusals = []
    for space in spaces_for(info):
        extra = EXTRA.get(name)
        code = f"trajectory = {name}({space}{', ' + extra if extra else ''})"
        scope = {**namespace, name: info.function}
        started = time.perf_counter()
        try:
            exec(compile(code, f"<card {name}>", "exec"), scope)
        except Exception as error:  # noqa: BLE001 - a refusal is the card's content
            refusals.append((space, code, f"{type(error).__name__}: {error}"))
            continue
        return {"space": space, "code": code, "trajectory": scope["trajectory"],
                "seconds": time.perf_counter() - started, "refusals": refusals}
    return {"space": None, "code": refusals[-1][1], "trajectory": None, "refusals": refusals}


def _style(axes, ylabel: str, xlabel: bool) -> None:
    if xlabel:
        axes.set_xlabel("time (s)", fontsize=7.5, color="#52514e", labelpad=1)
    axes.set_ylabel(ylabel, fontsize=7.5, color="#52514e", labelpad=2)
    axes.tick_params(labelsize=7, colors="#52514e", length=2.5, width=0.6)
    axes.grid(True, color="#e6e4df", linewidth=0.6)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#c9c6bf")
        axes.spines[side].set_linewidth(0.7)


def plot_kind(name: str, trajectory, effort: bool) -> tuple[str, int]:
    """Joint positions in degrees against time; an effort-commanded kind adds its command (N·m) as a second panel
    on the same clock (WHY two panels, never a second y-axis: the dataviz rule, and its positions are only predicted)."""

    t = np.asarray(trajectory.t, float)
    step = max(1, math.ceil(len(t) / PLOT_POINTS))
    panels = [("predicted position (°)" if effort else "position (°)", np.degrees(np.asarray(trajectory.positions, float)))]
    if effort and trajectory.effort is not None:
        panels.append(("effort (N·m)", np.asarray(trajectory.effort, float)))
    height = 1.85 if len(panels) == 1 else 3.1
    figure, axes_list = plt.subplots(len(panels), 1, figsize=(4.6, height), dpi=PLOT_DPI, sharex=True, squeeze=False)
    names = [joint.removesuffix("_joint") for joint in trajectory.joint_names]
    for row, ((ylabel, values), axes) in enumerate(zip(panels, axes_list[:, 0])):
        for index in range(values.shape[1]):
            axes.plot(t[::step], values[::step, index], color=JOINT_COLOURS[index % len(JOINT_COLOURS)], linewidth=1.2,
                      label=names[index])
        _style(axes, ylabel, xlabel=row == len(panels) - 1)
    axes = axes_list[0, 0]
    axes.set_xlim(t[0], t[-1] if t[-1] > t[0] else t[0] + 1.0)
    if len(names) > 1:
        axes.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(names), fontsize=6.4, frameon=False,
                    handlelength=1.2, columnspacing=0.9, borderaxespad=0.1, handletextpad=0.4)
    figure.tight_layout(pad=0.35)
    file_name = f"{name}.png"
    figure.savefig(IMG / file_name, dpi=PLOT_DPI, facecolor="white")
    plt.close(figure)
    # WHY a fixed-palette PNG (the joint colours blended into white, plus greys): 77 plots load in a third of the bytes,
    # and every joint keeps its exact colour (an adaptive palette merged orange into yellow)
    with Image.open(IMG / file_name) as picture:
        picture.convert("RGB").quantize(palette=PLOT_PALETTE, dither=Image.Dither.NONE).save(IMG / file_name, optimize=True)
    return file_name, round(height * PLOT_DPI)


def plot_palette() -> Image.Image:
    colours = [tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)) for hex_colour in JOINT_COLOURS]
    entries = [tuple(round(255 - (255 - c) * k / 8) for c in colour) for colour in colours for k in range(1, 9)]
    entries += [(round(255 * k / 31),) * 3 for k in range(32)]
    entries += [(255, 255, 255)] * (256 - len(entries))
    palette = Image.new("P", (1, 1))
    palette.putpalette([value for entry in entries for value in entry])
    return palette


PLOT_PALETTE = plot_palette()


def scrub(text: str) -> str:
    """A refusal as the reader would see it, without the builder's own paths."""
    return text.replace(str(ROOT) + "/", "").replace(str(ROOT), ".")


def kind_card(name: str, info, run: dict) -> str:
    tags = "".join(f"<span class='tag' title='{esc(USE_CASES.get(tag, ''))}'>{esc(words(tag))}</span>" for tag in info.use_cases)
    extra_needs = [need for need in info.needs if need not in ("joint_limits", "velocity_limits", "acceleration_limits")]
    needs = ("joint limits" + "".join(f" + {words(need)}" for need in extra_needs))
    chips = []
    if info.higher_order:
        chips.append("<span class='chip'>higher-order</span>")
    if info.command == "effort":
        chips.append("<span class='chip end'>effort-commanded</span>")
    if info.plugin:
        chips.append(f"<span class='chip'>plugin {esc(info.plugin)}</span>")
    trajectory = run["trajectory"]
    if trajectory is not None:
        image, image_height = plot_kind(name, trajectory, info.command == "effort")
        segments = len(robot_trajectories.segment_info(trajectory))
        stand_in = " (stand-ins from <code>tests/fixtures</code>)" if run["space"] in STAND_IN else ""
        body = (f"<pre class='src call'>{python(run['code'])}</pre>"
                f"<img src='img/{image}' width='552' height='{image_height}' loading='lazy' "
                f"alt='{esc(name)}: joint positions in degrees against time'>"
                f"<p class='meta'>{float(trajectory.t[-1]):.1f} s · {len(trajectory.t):,} samples · {segments} "
                f"segment{'s' if segments != 1 else ''} · on <b>{esc(SPACE_ABOUT[run['space']])}</b>{stand_in}</p>")
    else:
        last = run["refusals"][-1]
        body = (f"<pre class='src call'>{python(run['code'])}</pre>"
                f"<div class='refused'><b>No plot: it needs what this build does not have.</b> On "
                f"<code>{esc(last[0])}</code> it raised:<pre class='out wrap'>{esc(scrub(last[2]))}</pre></div>")
    catalog_row = f" · row <code>{esc(info.catalog_id)}</code>" if info.catalog_id and info.catalog_id != name else ""
    return (f"<article class='card' id='kind-{esc(name)}'><header><h4>{esc(name)}</h4>{''.join(chips)}</header>"
            f"<p class='desc'>{prose(info.one_line)}</p>"
            f"<p class='needs'><span class='muted'>needs</span> {esc(needs)}{catalog_row}</p>"
            f"<div class='tags'>{tags}</div>{body}</article>")


def family_blurbs() -> dict[str, str]:
    """Each family's one-line meaning: the comment beside it in trajectory_catalog/kind.py's FAMILIES."""
    source = Path(inspect.getsourcefile(vocabulary)).read_text(encoding="utf-8")
    block = source.split("FAMILIES: tuple[str, ...] = (", 1)[1].split("\n)", 1)[0]
    return {match.group(1): match.group(2).strip() for match in re.finditer(r'"(\w+)",\s*#\s*(.+)', block)}


# ---- the API table: names and their first docstring line, from the package ----------------------------------------

API = (
    ("the robot", ("JointSpace",)),
    ("making kinds", ("kind", "kinds", "catalog")),
    ("composing", ("chain", "split", "repeat", "annotate")),
    ("the record", ("recipe", "rebuild", "summary")),
    ("timing legs", ("WaypointTrajectoryGenerator", "waypoint_trajectory_generators", "TrajectoryGenerator")),
)


def api_table() -> str:
    rows = []
    for group, names in API:
        for index, name in enumerate(names):
            group_cell = f"<td class='for' rowspan='{len(names)}'><b>{group}</b></td>" if index == 0 else ""
            rows.append(f"<tr>{group_cell}<td class='name'><code>{name}</code></td>"
                        f"<td class='what'>{prose(first_line(getattr(robot_trajectories, name)))}</td></tr>")
    return (f"<div class='scroll'><table class='api'><thead><tr><th>for</th><th>name</th><th>what it is</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def on_pypi() -> bool:
    """Whether the package is published, so the page never shows an install line that fails."""
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/robot-trajectories/json", timeout=10) as reply:
            return reply.status == 200
    except Exception:  # noqa: BLE001 - offline or 404: not published (as far as this build can tell)
        return False


def plugin_entry_points() -> str:
    text = (ROOT / "plugins" / "robot_trajectories_ruckig" / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"\[project\.entry-points\.[^\n]+\n(?:[^\n\[]+\n)+", text).group(0).strip()
    return highlight(block, TOMLLexer(), FORMATTER).rstrip("\n")


CSS = """
:root { --bg:#fbfaf7; --panel:#ffffff; --ink:#1d1f23; --muted:#5d6470; --line:#e3e0d8; --accent:#2f6f4f;
  --accent-soft:#e8f2ec; --warn:#9a5b00; --warn-soft:#fbf0dc; --code:#f3f1ec; --out-bg:#f3f1ec;
  --out-ink:#4f5661; --out-rule:#d3cec2; --paper:#ffffff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#15171a; --panel:#1d2024; --ink:#e7e5e0;
  --muted:#a3a9b2; --line:#33373d; --accent:#7cc39b; --accent-soft:#1f3329; --warn:#f0b35a; --warn-soft:#3a2d17;
  --code:#24272c; --out-bg:#17191c; --out-ink:#a3a9b2; --out-rule:#3d4249; --paper:#f4f2ee; } }
:root[data-theme="dark"] { --bg:#15171a; --panel:#1d2024; --ink:#e7e5e0; --muted:#a3a9b2; --line:#33373d;
  --accent:#7cc39b; --accent-soft:#1f3329; --warn:#f0b35a; --warn-soft:#3a2d17; --code:#24272c;
  --out-bg:#17191c; --out-ink:#a3a9b2; --out-rule:#3d4249; --paper:#f4f2ee; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; background:var(--bg); color:var(--ink); font:15.5px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }
a { color:var(--accent); }
.wrap { max-width:1200px; margin:0 auto; padding:0 16px; }
header.hero { padding:26px 0 14px; border-bottom:1px solid var(--line); }
.eyebrow { color:var(--muted); font-size:13px; letter-spacing:.02em; margin:0 0 6px; }
h1 { font-size:32px; margin:0 0 4px; letter-spacing:-.01em; }
h1 code { background:none; padding:0; font-size:inherit; }
.lede { font-size:17px; max-width:900px; margin:0 0 10px; }
.links { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 0; }
.links a { text-decoration:none; border:1px solid var(--line); background:var(--panel); padding:5px 12px;
  border-radius:999px; font-size:14px; }
.install { margin:12px 0 0; max-width:760px; }
.install pre { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:9px 12px; overflow-x:auto; margin:4px 0; }
.install p { margin:0; font-size:13.5px; }
nav.toc { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); }
nav.toc .wrap { display:flex; gap:18px; overflow-x:auto; white-space:nowrap; padding-top:9px; padding-bottom:9px; }
nav.toc a { text-decoration:none; color:var(--muted); font-size:14px; } nav.toc a:hover { color:var(--ink); }
section { padding:22px 0 8px; }
h2 { font-size:24px; margin:8px 0 6px; } h3 { font-size:18px; margin:26px 0 8px; } h4 { font-size:16px; margin:0; }
.section-lede { color:var(--muted); max-width:860px; margin:0 0 14px; }
.split { display:grid; grid-template-columns:minmax(0,5fr) minmax(0,7fr); gap:20px; align-items:start; }
.glance { display:grid; grid-template-columns:minmax(0,1fr); gap:10px; margin:2px 0 10px; }
.glance > div { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:10px 14px; }
.glance-code h2 { margin-top:0; }
.glance b.k { display:block; font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--accent); }
.glance p { margin:4px 0 0; font-size:14.5px; }
.cell-caption { color:var(--muted); margin:4px 0 8px; font-size:14.5px; }
.notebook { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; margin:6px 0 16px; }
.notebook pre { margin:0; border:0; border-radius:0; overflow-x:auto; max-width:100%; }
pre { font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre; }
pre.src { background:var(--panel); color:var(--ink); padding:8px 14px; }
pre.src.gap { padding-top:20px; }
pre.out { background:var(--out-bg); color:var(--out-ink); border-left:3px solid var(--out-rule);
  padding:5px 14px 6px 11px; font-size:12.2px; }
pre.out.wrap { white-space:pre-wrap; overflow-wrap:anywhere; }
code { background:var(--code); padding:1px 5px; border-radius:4px; font:12.8px ui-monospace,SFMono-Regular,Menlo,monospace; }
pre code { background:none; padding:0; }
table { border-collapse:collapse; width:100%; background:var(--panel); font-size:14px; }
th, td { border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
th { background:var(--code); font-weight:600; }
.scroll { overflow-x:auto; margin:8px 0 14px; border-radius:8px; }
.api td.for, .api td.name { white-space:nowrap; }
p code, li code, td.what code { overflow-wrap:anywhere; }
.two { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:18px; align-items:start; }
.counts { display:flex; flex-wrap:wrap; gap:6px; margin:0 0 6px; }
.counts a { text-decoration:none; font-size:13px; border:1px solid var(--line); background:var(--panel); border-radius:999px;
  padding:2px 10px; color:var(--ink); } .counts a span { color:var(--muted); }
.group { margin:26px 0 6px; } .group h3 { margin:0 0 2px; } .group > p { color:var(--muted); margin:0 0 10px; }
.cards { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin-top:10px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 14px; min-width:0;
  display:flex; flex-direction:column; }
.card header { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin-bottom:2px; }
.card header h4 { font:600 15px ui-monospace,SFMono-Regular,Menlo,monospace; margin-right:auto; overflow-wrap:anywhere; }
.card .desc { margin:2px 0 6px; font-size:14px; }
.card .needs { margin:0 0 4px; font-size:13px; }
.card .tags { margin:0 0 8px; }
.card pre.call { background:var(--code); border-radius:6px; padding:6px 10px; margin:auto 0 8px; font-size:12px;
  white-space:pre-wrap; overflow-wrap:anywhere; }
.card img { width:100%; height:auto; display:block; background:var(--paper); border:1px solid var(--line); border-radius:8px; }
.card .meta { color:var(--muted); font-size:12.5px; margin:6px 0 0; }
.card .meta code { font-size:11.5px; }
.refused { background:var(--warn-soft); border-radius:8px; padding:8px 10px; font-size:13px; }
.refused pre.out { margin:6px 0 0; background:transparent; border-left-color:var(--warn); }
.tag { display:inline-block; font-size:12px; background:var(--code); border-radius:6px; padding:1px 7px; margin:2px 2px 0 0; }
.chip { font-size:11.5px; border-radius:999px; padding:1px 8px; background:var(--code); color:var(--muted); }
.chip.end { background:var(--warn-soft); color:var(--warn); }
.muted { color:var(--muted); }
.seam pre { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:9px 12px; overflow-x:auto; }
.seam ul { padding-left:20px; } .seam li { margin:4px 0; }
footer { border-top:1px solid var(--line); margin-top:30px; padding:18px 0 40px; color:var(--muted); font-size:13.5px; }
@media (max-width: 1000px) { .cards { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width: 900px) { .split, .two { grid-template-columns:minmax(0,1fr); } }
@media (max-width: 640px) {
  h1 { font-size:28px; } .lede { font-size:16.5px; }
  .cards { grid-template-columns:minmax(0,1fr); } .card { padding:12px; }
  .api thead { display:none; } .api tr, .api td { display:block; border:0; } .api td.for { padding:10px 8px 0; }
  .api td.name { padding:6px 8px 0; } .api td.what { padding:2px 8px 8px; border-bottom:1px solid var(--line); }
}
"""


# ---- the build -----------------------------------------------------------------------------------------------------

def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    IMG.mkdir(parents=True)

    quick_start = without_docstring((ROOT / "examples" / "quick_start.py").read_text(encoding="utf-8"))
    glance = run_script("site-glance", [("examples/quick_start.py", "The CAN 11 pendulum from bare joint limits: a "
                                         "higher-order kind, chained, split back into pieces and rebuilt from its "
                                         "recipe.", quick_start)])
    spaces = run_script("site-spaces", [("The spaces the cards run on", "", SPACES_CODE)])
    names = run_script("site-names", [("Two names", "", NAMES_CODE)])

    registry = robot_trajectories.kinds()
    blurbs = family_blurbs()
    by_family: dict[str, list[str]] = {family: [] for family in vocabulary.FAMILIES}
    for name, info in registry.items():
        by_family.setdefault(info.family, []).append(name)
    plotted, refused, groups, counts = 0, [], [], []
    for family, members in by_family.items():
        if not members:
            continue
        cards = []
        for name in sorted(members):
            started = time.perf_counter()
            run = run_kind(name, registry[name], spaces.namespace)
            cards.append(kind_card(name, registry[name], run))
            if run["trajectory"] is None:
                refused.append(name)
            else:
                plotted += 1
            print(f"  card {name}: {run['space'] or 'refused'} · {time.perf_counter() - started:.1f} s", flush=True)
        counts.append(f"<a href='#family-{family}'>{esc(words(family))} <span>{len(members)}</span></a>")
        groups.append(f"<div class='group' id='family-{family}'><h3>{esc(words(family))} <span class='muted'>· "
                      f"{len(members)}</span></h3><p>{esc(blurbs.get(family, ''))}</p>"
                      f"<div class='cards'>{''.join(cards)}</div></div>")
    todo = robot_trajectories.catalog().todo
    todo_html = (f"<p class='section-lede'>Catalogued, not generated yet: {esc(', '.join(todo.names))}.</p>"
                 if len(todo) else "<p class='section-lede'>Every catalog row is generated by a kind.</p>")

    published = on_pypi()
    if published:
        install = (f"<pre>pip install robot-trajectories</pre><p class='muted'>The runtime needs numpy and robot_msgs. "
                   f"Ruckig and TOPP-RA timing come as plugins in <code>plugins/</code>.</p>")
        pypi_link = f"<a href='{PYPI}'>PyPI</a>"
    else:
        install = (f"<pre>pip install \"robot-trajectories @ git+{REPO}\"</pre><p class='muted'>Not on PyPI yet, so "
                   f"it installs from GitHub; robot_msgs comes from PyPI. The runtime needs numpy and robot_msgs; Ruckig "
                   f"and TOPP-RA timing come as plugins in <code>plugins/</code>.</p>")
        pypi_link = ""

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>robot_trajectories</title>
<meta name="description" content="A signal library for robot motion: every trajectory kind is a plain function of a
JointSpace that returns a robot_msgs JointTrajectory inside the limits.">
<style>{CSS}
{token_css()}</style></head>
<body>
<header class="hero"><div class="wrap">
  <p class="eyebrow">v{VERSION} · MIT · Python ≥ 3.11 · runtime: numpy + robot_msgs</p>
  <h1><code>robot_trajectories</code></h1>
  <p class="lede">A signal library for robot motion. Every kind of trajectory is a plain function: pass a
  <code>JointSpace</code> (the robot) and numbers, points or poses (the task), get back one robot_msgs
  <code>JointTrajectory</code>, inside the limits and at rest at both ends, carrying the call that made it.
  {len(registry)} kinds; <code>chain</code> and <code>split</code> compose them.</p>
  <div class="links"><a href="{REPO}">GitHub</a>{pypi_link}<a href="{PEP}">Design (PEP 0002)</a>
  <a href="#catalog">Catalog</a><a href="#names">Names</a></div>
  <div class="install">{install}</div>
</div></header>
<nav class="toc"><div class="wrap"><a href="#glance">At a glance</a><a href="#api">API</a><a href="#catalog">Catalog</a>
<a href="#names">Names</a><a href="#plugins">Plugins</a></div></nav>
<main class="wrap">

<section id="glance">
  <div class="split">
    <div class="glance">
      <div><b class="k">What goes in</b><p>The robot as a <code>JointSpace</code>: joint limits first, then
        kinematics (FK/IK), dynamics, a collision scene, each injected by shape (<code>with_kinematics</code>,
        <code>with_dynamics</code>, <code>with_collision</code>), never imported. The task as points, poses and
        numbers.</p></div>
      <div><b class="k">What a kind is</b><p><code>f(space, ..., start=None) -&gt; JointTrajectory</code>, registered
        with <code>@kind</code>. It knows joints, poses, limits and time, never what the motion is for: no task words
        in a kind's name, parameters or labels. What it serves is a use-case tag.</p></div>
      <div><b class="k">What comes out</b><p>One robot_msgs <code>JointTrajectory</code>, guarded inside the limits,
        with its segments labelled and its recipe recorded: <code>rebuild(recipe(t))</code> makes it again, sample for
        sample. A kind that needs a rung the space lacks raises <code>MissingRung</code>, naming the call that adds
        it.</p></div>
      <div><b class="k">Units</b><p>SI in code (rad, rad/s, m); degrees on every display: summaries,
        <code>describe()</code>, and every plot on this page.</p></div>
    </div>
    <div class="glance-code"><h2>{esc(glance.cells[0]['label'])}</h2>
      <p class="cell-caption">{esc(glance.cells[0]['caption'])}</p>{notebook(glance.cells[0], glance.script)}</div>
  </div>
  <div id="api"><h3>The API in one table</h3>{api_table()}</div>
</section>

<section id="catalog">
  <h2>Catalog <span class="muted">· {len(registry)} kinds, {plotted} plotted here</span></h2>
  <p class="section-lede">Every registered kind (<code>robot_trajectories.kinds()</code>), grouped by family. Each card
  is one call, run by this page's builder with the kind's defaults on the simplest of the four spaces below that it runs
  on, and a plot of the joint positions it returned, in degrees. Hover a tag for its use case.</p>
  <div class="counts">{''.join(counts)}</div>
  <details><summary>The four spaces the cards run on</summary>{notebook(spaces.cells[0], spaces.script)}</details>
  {''.join(groups)}
  {todo_html}
</section>

<section id="names">
  <h2>Names</h2>
  <div class="two">
    <div><p class="section-lede">The stage is <i>trajectory generation</i>. A <b><code>TrajectoryGenerator</code></b>
      is anything that hands back a robot_msgs <code>Trajectory</code> when called: every kind is one. A
      <b><code>WaypointTrajectoryGenerator</code></b> is the one kinds hand their legs' timing to: points and joint
      limits in, a <code>JointTrajectory</code> out. Minimum-jerk and trapezoid are built in; Ruckig, Ruckig Pro and
      TOPP-RA come as plugins. A path kind says where the joints go, then hands each leg to the generator passed as
      <code>waypoint_generator=</code>.</p>
      <p class="muted">Formerly <code>TimeLaw</code>, then <code>TimeParameterization</code>: the old names still import
      as the same objects, the old keywords still work, and recipes keep their <code>"time_law"</code> keys.</p></div>
    <div>{notebook(names.cells[0], names.script)}</div>
  </div>
</section>

<section id="plugins" class="seam">
  <h2>The plugin seam</h2>
  <p class="section-lede">Anything this leaf may not import (Ruckig, TOPP-RA, a robot's own kinds) registers itself in
  its own <code>pyproject.toml</code>, found by Python entry points, never named here. The Ruckig plugin's, as
  shipped in <code>plugins/robot_trajectories_ruckig</code>:</p>
  <pre>{plugin_entry_points()}</pre>
  <ul>
    <li><code>robot_trajectories.kinds</code>: a module whose <code>@kind</code> functions register on import; a
      plugin kind is listed, tagged, guarded and recorded like a built-in one.</li>
    <li><code>robot_trajectories.waypoint_trajectory_generators</code>: a backend that times points within limits;
      it then serves <code>WaypointTrajectoryGenerator(name, ...)</code>.</li>
  </ul>
  <p class="muted">This page was built with the Ruckig and TOPP-RA plugins installed, so the kinds that need them have
  plots. The page is built by <code>website/build_site.py</code>; every number and picture on it came from that run.</p>
</section>
</main>
<footer><div class="wrap">robot_trajectories v{VERSION} · MIT © 2026 Zach Yamaoka · built {time.strftime('%Y-%m-%d')}
from <a href="{REPO}">{esc(REPO.removeprefix('https://'))}</a> · the UR5e and the pendulum's dynamics are the test
suite's stand-ins, not measured models.</div></footer>
</body></html>
"""
    leaks = sorted({match.group(0) for match in PRIVATE.finditer(page)})
    if leaks:
        raise SystemExit(f"refusing to write: private material in the page: {leaks}")
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    size = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    print(f"wrote {OUT / 'index.html'} · {plotted} kinds plotted, {len(refused)} without a plot "
          f"({', '.join(refused) or 'none'}) · {len(list(IMG.glob('*.png')))} images · {size / 1e6:.2f} MB in all")


if __name__ == "__main__":
    main()
