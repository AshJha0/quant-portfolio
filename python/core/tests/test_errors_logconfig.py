"""qp_core.errors (W8) and qp_core.logconfig (W7)."""

from __future__ import annotations

import io
import json
import logging

import pytest
import qp_core
from qp_core import logconfig
from qp_core.errors import ConvergenceError, DataError, InputError, QPError


@pytest.fixture(autouse=True)
def _clean_logging():
    yield
    logconfig.reset_logging()


class TestErrors:
    @pytest.mark.parametrize(
        ("cls", "builtin"),
        [(InputError, ValueError), (ConvergenceError, RuntimeError), (DataError, RuntimeError)],
    )
    def test_each_class_is_a_qperror_and_its_old_builtin(self, cls, builtin):
        assert issubclass(cls, QPError)
        assert issubclass(cls, builtin)
        with pytest.raises(builtin):
            raise cls("x")

    def test_the_three_kinds_are_disjoint(self):
        kinds = (InputError, ConvergenceError, DataError)
        for a in kinds:
            for b in kinds:
                assert (a is b) == issubclass(a, b)

    def test_convergence_error_is_still_a_valueerror(self):
        # Callers that caught ValueError around solver calls keep working.
        assert issubclass(ConvergenceError, ValueError)
        assert not issubclass(DataError, ValueError)

    def test_qperror_is_not_a_valueerror(self):
        # Only InputError widens to ValueError; the base stays neutral.
        assert not issubclass(QPError, ValueError)

    def test_public_names(self):
        from qp_core import errors

        assert sorted(errors.__all__) == ["ConvergenceError", "DataError", "InputError", "QPError"]


class TestLibraryLoggingIsSilent:
    def test_qp_core_has_only_a_null_handler(self):
        handlers = logging.getLogger("qp_core").handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.NullHandler)

    def test_import_does_not_touch_the_root_logger(self):
        assert not any(h.get_name() == "qp_core.logconfig" for h in logging.getLogger().handlers)


class TestConfigureLogging:
    def test_record_is_one_json_line_with_the_run_context(self):
        buf = io.StringIO()
        ctx = logconfig.configure_logging(stream=buf, run_id="r-1", git_commit="deadbeef", as_of="2026-11-03")
        logging.getLogger("qp.test").warning("hello %s", "world")
        lines = buf.getvalue().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["msg"] == "hello world"
        assert rec["level"] == "WARNING"
        assert rec["run_id"] == "r-1" == ctx.run_id
        assert rec["git_commit"] == "deadbeef"
        assert rec["version"] == qp_core.__version__
        assert rec["as_of"] == "2026-11-03"
        assert rec["ts"].endswith("+00:00")

    def test_default_level_is_warning(self):
        buf = io.StringIO()
        logconfig.configure_logging(stream=buf, git_commit="x")
        logging.getLogger("qp.test").info("quiet")
        assert buf.getvalue() == ""

    def test_reconfiguring_replaces_rather_than_stacks(self):
        a, b = io.StringIO(), io.StringIO()
        logconfig.configure_logging(stream=a, git_commit="x")
        logconfig.configure_logging(stream=b, git_commit="x")
        logging.getLogger("qp.test").warning("once")
        assert a.getvalue() == ""
        assert len(b.getvalue().splitlines()) == 1

    def test_run_id_defaults_to_a_fresh_uuid(self):
        r1 = logconfig.configure_logging(stream=io.StringIO(), git_commit="x").run_id
        r2 = logconfig.configure_logging(stream=io.StringIO(), git_commit="x").run_id
        assert r1 != r2 and len(r1) == 32

    def test_git_commit_is_found_or_unknown(self):
        ctx = logconfig.configure_logging(stream=io.StringIO())
        assert ctx.git_commit == "unknown" or len(ctx.git_commit) == 40

    def test_reset_clears_the_context(self):
        logconfig.configure_logging(stream=io.StringIO(), git_commit="x")
        assert logconfig.current_context() is not None
        logconfig.reset_logging()
        assert logconfig.current_context() is None


class TestRunBoundary:
    def test_failure_leaves_one_record_naming_input_as_of_and_error_type(self):
        buf = io.StringIO()
        logconfig.configure_logging(stream=buf, git_commit="x", as_of="2026-11-03")
        with pytest.raises(DataError), logconfig.run_boundary("eod-var", input="books/fx_book.json"):
            raise DataError("EURUSD fixing missing for 2026-11-03")
        lines = buf.getvalue().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["level"] == "ERROR"
        assert rec["input"] == "books/fx_book.json"
        assert rec["as_of"] == "2026-11-03"
        assert rec["error_type"] == "qp_core.errors.DataError"
        assert "EURUSD fixing missing" in rec["msg"]
        assert "Traceback" in rec["exception"]

    def test_success_logs_nothing(self):
        buf = io.StringIO()
        logconfig.configure_logging(stream=buf, git_commit="x")
        with logconfig.run_boundary("eod-var"):
            pass
        assert buf.getvalue() == ""

    def test_keyboard_interrupt_is_not_logged(self):
        buf = io.StringIO()
        logconfig.configure_logging(stream=buf, git_commit="x")
        with pytest.raises(KeyboardInterrupt), logconfig.run_boundary("eod-var"):
            raise KeyboardInterrupt
        assert buf.getvalue() == ""
