"""Every kind of trajectory, one record class per file. Importing this package imports every module in its folder.

WHY discovery instead of an import list (V2): adding a kind is one new file here and nothing else (no registry edit, no
export line). The cost: the names are bound at import time, so a static checker sees them only through the modules.
"""

from __future__ import annotations

import importlib
import pkgutil

from ..kind import catalog

for _module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module.name}")

globals().update({kind.__name__: kind for kind in catalog()})
__all__ = [kind.__name__ for kind in catalog()]
