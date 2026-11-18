"""Every kind of trajectory, one function per file. Importing this package imports every module in its folder, and each
module's ``@kind`` registers its function (``robot_trajectories.kinds()`` is the registry).

WHY discovery instead of an import list (V1 and V2 both): adding a kind is one new file here and nothing else (no registry
edit, no export line). A kind another package provides registers the same way through the ``robot_trajectories.kinds``
entry-point group (``plugins.py``). A module whose name starts with ``_`` is a private helper a lane shares between its own
kinds (KINDS_TODO.md); it is imported too, and registers nothing.
"""

from __future__ import annotations

import importlib
import pkgutil

for _module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module.name}")
