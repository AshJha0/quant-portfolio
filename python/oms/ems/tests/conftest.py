"""Make python/oms importable as ``qp_oms`` until the r12-l2-oms package lands.

Integration seam: once python/oms/core ships the ``qp_oms`` distribution with
these folders as subpackages, delete this file.
"""

import importlib
import sys
import types
from pathlib import Path

_root = str(Path(__file__).resolve().parents[2])
try:
    _mod = importlib.import_module("qp_oms")
except ImportError:
    _mod = types.ModuleType("qp_oms")
    _mod.__path__ = []  # type: ignore[attr-defined]
    sys.modules["qp_oms"] = _mod
if _root not in list(_mod.__path__):
    _mod.__path__.append(_root)
