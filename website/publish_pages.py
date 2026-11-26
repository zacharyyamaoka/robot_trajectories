"""Publish website/_build/ to the gh-pages branch (GitHub Pages serves it) and push it.

    python3 /home/bam/robot_trajectories/website/publish_pages.py      # after website/build_site.py (its docstring has the build)

WHY a build artifact on its own branch (robot_metrics' pattern, 2026-11-18): the page is built with matplotlib, pygments and
the Ruckig / TOPP-RA plugins, none of which this leaf may depend on; its images stay out of main's history.
The commit is made with a temporary index (never the shared one) and never touches the working tree: one new commit on top of gh-pages, named after the main commit it was built from.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "website" / "_build"
BRANCH = "gh-pages"


def git(*args: str, env: dict | None = None) -> str:
    done = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, env=env)
    if done.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def main() -> None:
    if not (BUILD / "index.html").is_file():
        raise SystemExit(f"nothing to publish: build it first ({BUILD / 'index.html'} is missing)")
    source = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "--", "src", "website", "docs", "examples", "tests", "plugins",
                "pyproject.toml")
    with tempfile.TemporaryDirectory() as scratch:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch) / "index")}
        git("--work-tree", str(BUILD), "add", "--all", "--force", ".", env=env)
        tree = git("write-tree", env=env)
    parent = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", f"refs/heads/{BRANCH}"],
                            capture_output=True, text=True).stdout.strip()
    if parent and git("rev-parse", f"{parent}^{{tree}}") == tree:
        print(f"{BRANCH} already holds this build ({parent[:9]})")
    else:
        message = f"site: built from main {source}" + (" (with uncommitted changes)" if dirty else "")
        commit = git("commit-tree", tree, *(["-p", parent] if parent else []), "-m", message)
        git("update-ref", f"refs/heads/{BRANCH}", commit, *([parent] if parent else []))
        print(f"{BRANCH} -> {commit[:9]}: {message}")
    if "--no-push" not in sys.argv:
        print(git("push", "origin", f"{BRANCH}:{BRANCH}"))


if __name__ == "__main__":
    main()
