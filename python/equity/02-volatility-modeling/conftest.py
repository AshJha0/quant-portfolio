"""Ensure ``src`` layout is importable when running pytest from the project root
without an editable install (tests must pass offline)."""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
