"""The example pipeline runs on a base install and writes nothing unless asked."""
import subprocess
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1] / "examples" / "run_pipeline.py"

_PROBE = (
    "import runpy, sys\n"
    "sys.modules['matplotlib'] = None  # any import of matplotlib now fails\n"
    "sys.argv = [sys.argv[1]] + sys.argv[2:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


def _snapshot(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if "__pycache__" not in p.parts}


def test_pipeline_runs_without_matplotlib_and_writes_nothing(tmp_path):
    project = PIPELINE.parents[1]
    before = _snapshot(project)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(PIPELINE)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "VALIDATION REPORT" in proc.stdout
    assert _snapshot(project) == before
    assert list(tmp_path.iterdir()) == []


def test_pipeline_out_writes_report_only_under_out(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(PIPELINE), "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "report.txt").read_text(encoding="utf-8").startswith("=")
    assert not (tmp_path / "figures").exists()
