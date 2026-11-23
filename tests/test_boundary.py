"""The leaf rule: the package imports only the standard library, numpy and robot_msgs, and holds no rig's values.

    /home/bam/robot_trajectories/.venv/bin/python /home/bam/robot_trajectories/tests/test_boundary.py

robot_metrics' approach (tests/test_import_isolation.py there): an AST check over every file, held red by planted violations
so a checker that passes everything cannot pass this file; an import check in a fresh interpreter (importing every module of
the package loads no third-party module that robot_msgs itself does not); and the same import in an isolated uv environment
(``env -u VIRTUAL_ENV uv run --isolated``), whose env holds only what pyproject.toml declares.

WHY (Zach, "dependency-free runtime cores"; the 2026-11-18 round's contract): a RobotDescription, a ReachMap, kinematics, a
dynamics model or a collision checker enter by shape, never by import, so sim2real, apps and rigs can depend on this package and
it on none of them. Plugins (Ruckig, TOPP-RA, a robot's own kinds) are found by entry point (``plugins.py``), never named here.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import _runner

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "robot_trajectories"
CORE_ROOTS = {"numpy", "robot_msgs", "robot_trajectories"}
NEVER = ("bam_reach", "bam_robot_descriptions", "bam_traj_gen", "bam_rack_twin", "pin_utils", "sim2real", "robot_models",
         "robot_metrics", "robot_controllers", "traj_compare", "pinocchio", "mujoco", "ruckig", "toppra", "scipy", "torch",
         "pandas", "matplotlib", "kernels", "mini_catalog", "robots", "fixtures", "scenario")
# A rig's own numbers belong to the rig (examples/, tests/fixtures/): the CAN 11 station's acceleration limit and its origin.
INSTANCE_VALUES = ("7.3896", "0.373685", "schedule.json")


def _imports(source: str) -> list[str]:
    """The root module of every absolute import in a file (relative imports stay inside the package)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append(node.module.split(".")[0])
    return found


def violations(src: Path = SRC) -> list[str]:
    bad = []
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(src).as_posix()
        text = path.read_text(encoding="utf-8")
        for root in _imports(text):
            if root in NEVER or not (root in sys.stdlib_module_names or root == "__future__" or root in CORE_ROOTS):
                bad.append(f"{rel}: imports {root}")
        for value in INSTANCE_VALUES:
            if value in text:
                bad.append(f"{rel}: holds the instance value {value}")
    return bad


def test_package_imports_only_its_layer():
    assert violations() == [], violations()


def _planted(folder: Path, rel: str, text: str) -> Path:
    root = folder / "robot_trajectories"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")
    return root


PLANTED = [
    ("registry.py", "import bam_reach\n", "registry.py: imports bam_reach"),
    ("space.py", "from bam_robot_descriptions import RobotDescription\n", "space.py: imports bam_robot_descriptions"),
    ("kinds/chirp.py", "def f():\n    import scipy.signal\n", "kinds/chirp.py: imports scipy"),
    ("segments.py", "import kernels as k\n", "segments.py: imports kernels"),
    ("time_law.py", "import ruckig\n", "time_law.py: imports ruckig"),
    ("this_week.py", "MAX_ACCELERATION = 7.3896\n", "this_week.py: holds the instance value 7.3896"),
    ("plugins.py", "from importlib import metadata\nfrom . import kernels as k\nimport numpy as np\nfrom robot_msgs import save_npz\n", None),
]


def test_checker_goes_red_on_a_planted_violation():
    for rel, text, expected in PLANTED:
        with tempfile.TemporaryDirectory() as folder:
            found = violations(_planted(Path(folder), rel, text))
            assert found == ([] if expected is None else [expected]), (rel, found)


def _import_all() -> str:
    """A script that imports every module of the package and prints the third-party roots that loaded beyond robot_msgs'."""
    return (
        "import sys\n"
        "import robot_msgs, robot_msgs.trajectory\n"
        "before = {name.split('.')[0] for name in sys.modules}\n"
        "import importlib, pkgutil, robot_trajectories\n"
        "for module in pkgutil.walk_packages(robot_trajectories.__path__, 'robot_trajectories.'):\n"
        "    importlib.import_module(module.name)\n"
        "after = {name.split('.')[0] for name in sys.modules}\n"
        "new = sorted(root for root in after - before if root not in sys.stdlib_module_names and root not in "
        f"{sorted(CORE_ROOTS)!r} and not root.startswith('_'))\n"
        f"never = sorted(name for name in {NEVER!r} if name in sys.modules)\n"
        "assert not new and not never, (new, never)\n"
        "print('robot_trajectories', robot_trajectories.__version__, 'imports clean')\n"
    )


def test_importing_every_module_loads_nothing_beyond_numpy_and_robot_msgs():
    done = subprocess.run([sys.executable, "-c", _import_all()], capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert done.returncode == 0 and "imports clean" in done.stdout, done.stderr[-2000:]


def test_core_imports_in_an_isolated_environment():
    if shutil.which("uv") is None:
        _runner.skip("uv is not installed")
    # WHY clear VIRTUAL_ENV (bam_ws CLAUDE.md): with it set, uv may reuse the active env and both red and green would lie.
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    env.update(UV_CONCURRENT_BUILDS="2", UV_CONCURRENT_INSTALLS="2", UV_CONCURRENT_DOWNLOADS="2")
    done = subprocess.run(["uv", "run", "--isolated", "--no-sources", "--project", str(ROOT), "python", "-c", _import_all()],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "imports clean" in done.stdout, done.stdout[-2000:]


if __name__ == "__main__":
    raise SystemExit(_runner.run(globals(), "test_boundary (the leaf rule)"))
