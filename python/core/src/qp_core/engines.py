"""Engine selection for the packages that have compiled ports.

The four packages with C++ and Rust ports (equity and FX options, equity and
FX VaR) each ship an ``engine`` module whose functions take
``engine="python" | "cpp" | "rust"``. The Python package is the reference;
``"cpp"`` calls the nanobind extension ``qp_cpp`` and ``"rust"`` the PyO3
extension ``qp_rust``. This module holds the one shared rule: how a name maps
to a backend and what happens when it cannot.

- An unknown name raises ``ValueError``.
- A known compiled engine whose extension is not installed (or did not
  import) raises ``ImportError`` with the command that installs it.

Results agree with the Python reference to the cross-language tolerances
stated in ``bindings/*/tests/test_binding.py`` and checked over 10,000
random inputs per engine in ``tests/test_engine_differential.py``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Literal

Engine = Literal["python", "cpp", "rust"]

ENGINES: tuple[str, ...] = ("python", "cpp", "rust")

_MODULES = {"cpp": "qp_cpp", "rust": "qp_rust"}
_INSTALL = {
    "cpp": "pip install ./bindings/cpp-nanobind  (C++23 compiler and CMake >= 3.26)",
    "rust": "pip install ./bindings/rust-pyo3  (Rust toolchain; maturin is fetched by pip)",
}


def check_engine(engine: str) -> str:
    """Return ``engine`` if it is a known name, else raise ``ValueError``."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected one of {', '.join(ENGINES)}")
    return engine


def load_engine(engine: str) -> ModuleType | None:
    """Return the compiled extension for ``engine``, or None for ``"python"``.

    Raises ``ValueError`` for an unknown name and ``ImportError`` (with an
    install hint) when the extension is missing or its build is broken.
    """
    check_engine(engine)
    if engine == "python":
        return None
    name = _MODULES[engine]
    hint = f"engine={engine!r} needs the {name} extension; from the repository root run: {_INSTALL[engine]}"
    try:
        mod = importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(hint) from exc
    if not mod.available():
        raise ImportError(hint)
    return mod


def engine_available(engine: str) -> bool:
    """True when ``engine`` can be used in this environment."""
    try:
        load_engine(engine)
    except ImportError:
        return False
    return True
