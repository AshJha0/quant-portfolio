"""Structured logging for application and study entry points.

The rule, from the Python logging HOWTO ("Configuring logging for a
library"): library code never configures handlers. Every quant-portfolio
package gets its logger with ``logging.getLogger(__name__)`` and leaves
handlers alone; ``qp_core`` attaches a single ``NullHandler`` so an
unconfigured program prints nothing. Library functions raise the typed
errors of :mod:`qp_core.errors` and do not log and re-raise.

Only an entry point (a CLI such as ``qp-run``, the EOD VaR runner, a study
driver) calls :func:`configure_logging`, once, and wraps its work in
:func:`run_boundary`. The boundary logs exactly one record when the run
fails, naming the input, the as-of date and the error type, then re-raises.

Each record is one JSON object per line carrying the run context:

``ts``          UTC time, ISO 8601 with milliseconds
``level``       ``WARNING``, ``ERROR``, ...
``logger``      logger name
``msg``         the formatted message
``run_id``      unique id of this run (uuid4 hex unless given)
``git_commit``  ``git rev-parse HEAD`` of the working tree, or ``"unknown"``
``version``     ``qp_core.__version__``
``as_of``       the business date the run is for, if any
``error_type``  qualified exception class, on records with an exception
``input``       what the run was reading, when the boundary knows it

Field names follow the OpenTelemetry log data model where one exists
(``error_type`` is its ``exception.type``). The default level is WARNING,
so routine runs stay quiet.

>>> import io, json, logging
>>> buf = io.StringIO()
>>> ctx = configure_logging(stream=buf, run_id="r1", git_commit="abc", as_of="2026-11-03")
>>> logging.getLogger("qp.demo").warning("curve has %d pillars", 3)
>>> rec = json.loads(buf.getvalue())
>>> (rec["msg"], rec["run_id"], rec["as_of"], rec["git_commit"])
('curve has 3 pillars', 'r1', '2026-11-03', 'abc')
>>> reset_logging()
"""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
import sys
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

__all__ = [
    "JsonFormatter",
    "RunContext",
    "configure_logging",
    "current_context",
    "reset_logging",
    "run_boundary",
]

_HANDLER_NAME = "qp_core.logconfig"
_context: RunContext | None = None


@dataclass(frozen=True)
class RunContext:
    """Identity of one run, stamped on every record it logs."""

    run_id: str
    git_commit: str
    version: str
    as_of: str | None = None
    extra: Mapping[str, str] = field(default_factory=dict)

    def as_fields(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra")
        d.update(extra)
        return d


def _git_commit(cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else "unknown"


class JsonFormatter(logging.Formatter):
    """One JSON object per record, with the active :class:`RunContext`."""

    def __init__(self, context: RunContext) -> None:
        super().__init__()
        self.context = context

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out.update(self.context.as_fields())
        for key in ("input", "error_type"):
            value = getattr(record, key, None)
            if value is not None:
                out[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            exc_type = record.exc_info[0]
            out.setdefault("error_type", f"{exc_type.__module__}.{exc_type.__qualname__}")
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str, sort_keys=False)


def configure_logging(
    *,
    level: int | str = logging.WARNING,
    stream: IO[str] | None = None,
    run_id: str | None = None,
    git_commit: str | None = None,
    as_of: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> RunContext:
    """Install the JSON handler on the root logger. Entry points only.

    Calling it again replaces the previous handler (it never stacks), so a
    test or a notebook can reconfigure safely. Returns the run context.
    """
    from qp_core import __version__

    global _context
    ctx = RunContext(
        run_id=run_id or uuid.uuid4().hex,
        git_commit=git_commit if git_commit is not None else _git_commit(),
        version=__version__,
        as_of=as_of,
        extra=dict(extra or {}),
    )
    reset_logging()
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(JsonFormatter(ctx))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
    _context = ctx
    return ctx


def reset_logging() -> None:
    """Remove the handler :func:`configure_logging` installed, if any."""
    global _context
    root = logging.getLogger()
    for h in list(root.handlers):
        if h.get_name() == _HANDLER_NAME:
            root.removeHandler(h)
            h.close()
    _context = None


def current_context() -> RunContext | None:
    """The active run context, or None when logging is not configured."""
    return _context


@contextlib.contextmanager
def run_boundary(
    name: str,
    *,
    input: str | None = None,
    logger: logging.Logger | None = None,
) -> Iterator[None]:
    """Log one ERROR record if the wrapped run fails, then re-raise.

    Use it once, around the whole body of an entry point::

        configure_logging(as_of="2026-11-03")
        with run_boundary("eod-var", input="books/fx_book.json"):
            run_eod_var(...)

    The record names the run, the input, the as-of date (from the context)
    and the exception type. Nothing is logged on success.
    """
    log = logger or logging.getLogger("qp.run")
    try:
        yield
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        log.error(
            "%s failed: %s",
            name,
            exc,
            exc_info=True,
            extra={
                "input": input,
                "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            },
        )
        raise
