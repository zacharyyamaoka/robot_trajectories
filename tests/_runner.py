"""Run a test file's ``test_*`` functions without pytest: ``python tests/test_x.py`` prints one line per test and a count.

WHY (the house rule while the arm is powered): pytest must not run on the rig's host, so every test file here is also a plain
script. Under pytest, ``skip`` raises pytest's own skip; as a script it raises ``Skipped`` and the runner counts it.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable, Mapping
from typing import Any, NoReturn


class Skipped(Exception):
    """A test whose input is absent (e.g. the round's media on another machine)."""


def skip(reason: str) -> NoReturn:
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(reason)
    raise Skipped(reason)


def run(namespace: Mapping[str, Any], name: str) -> int:
    """Every ``test_*`` function in ``namespace``, in definition order; returns the exit code (1 if any failed)."""
    tests: list[Callable[[], None]] = [value for key, value in namespace.items() if key.startswith("test_") and callable(value)]
    passed = failed = skipped = 0
    for test in tests:
        try:
            test()
        except Skipped as reason:
            skipped += 1
            print(f"  skip  {test.__name__}: {reason}")
        except AssertionError as error:
            failed += 1
            print(f"  FAIL  {test.__name__}: {str(error)[:2000]}")
        except Exception:  # noqa: BLE001 - an error is a failure, shown in full
            failed += 1
            print(f"  ERROR {test.__name__}:\n{traceback.format_exc()}")
        else:
            passed += 1
            print(f"  ok    {test.__name__}")
    verdict = "PASS" if not failed else "FAIL"
    print(f"{verdict}  {name}: {passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0
