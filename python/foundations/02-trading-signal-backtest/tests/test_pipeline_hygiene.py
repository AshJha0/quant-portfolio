"""The example pipeline must not write into the repository and must label
its data with the seed actually used."""

import importlib.util
from pathlib import Path

from eq_signal_backtest.data.synthetic import _DEFAULT_SEED

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_pipeline", ROOT / "examples" / "run_pipeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_output_dir_is_outside_the_repo():
    out = _load().default_out_dir().resolve()
    repo = ROOT.resolve().parents[2]
    assert repo not in out.parents and out != repo


def test_source_label_names_the_default_seed():
    assert _load().source_label(None) == f"bundled synthetic (seed={_DEFAULT_SEED})"
