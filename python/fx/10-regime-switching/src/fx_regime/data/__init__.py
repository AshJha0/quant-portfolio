"""Data generators for fx_regime (synthetic only; fully offline)."""

from .synthetic import (  # noqa: F401
    SyntheticPanel,
    generate_null_gbm_panel,
    generate_roro_panel,
)
