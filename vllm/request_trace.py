# SPDX-License-Identifier: Apache-2.0
"""Per-request, high-resolution timing trace for the request handling pipeline.

Each stage of a request's life cycle -- API preprocess, tokenize, engine
submit, scheduler queue/schedule, model forward, ... -- is timestamped with
``time.perf_counter_ns()``, which has nanosecond resolution and is monotonic
**within a single process**.

.. warning::

    Timestamps are **NOT comparable across processes**. vLLM runs the API
    frontend and the EngineCore in separate processes that communicate via
    msgpack. Each process must build its own :class:`RequestTrace` with its own
    ``t0``. Correlation across processes is done by ``request_id`` only --
    never by subtracting timestamps that were taken in different processes.

Enable by setting the ``VLLM_TRACE_REQUEST=1`` environment variable. When
disabled, :meth:`RequestTrace.mark` returns immediately at near-zero cost.
"""

from __future__ import annotations

import os
import time
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

from vllm.logger import init_logger

__all__ = [
    "RequestTrace",
    "current_request_trace",
    "mark_trace",
    "reset_current_request_trace",
    "set_current_request_trace",
    "trace_enabled",
]

if TYPE_CHECKING:
    pass

_logger = init_logger(__name__)

# Values that disable tracing. Everything else (e.g. "1", "true", "on") enables.
_DISABLED = {"", "0", "false", "False", "no", "off"}


def trace_enabled() -> bool:
    """Return whether request tracing is on.

    Read on every call so the flag can be toggled without restarting the
    process; the cost of a single ``os.environ`` lookup is negligible.
    """
    return os.environ.get("VLLM_TRACE_REQUEST", "0") not in _DISABLED


# Carries the trace of the request currently being served on this async task,
# *within the frontend process*. Lets deep call sites (the tokenizer in
# vllm/renderers/base.py, which has no request_id in scope) record stages
# without threading the trace object through every signature.
_current_request_trace: ContextVar[Optional["RequestTrace"]] = ContextVar(
    "vllm_current_request_trace", default=None
)


def current_request_trace() -> Optional["RequestTrace"]:
    """Return the trace bound to the current async context, if any."""
    return _current_request_trace.get()


def set_current_request_trace(trace: "RequestTrace"):
    """Bind ``trace`` to the current async context; returns a reset token."""
    return _current_request_trace.set(trace)


def reset_current_request_trace(token) -> None:
    """Restore the previous trace binding using the token from ``set_*``."""
    _current_request_trace.reset(token)


def mark_trace(trace: Optional["RequestTrace"], stage: str) -> None:
    """Record ``stage`` on ``trace`` if it is not None; otherwise a no-op.

    Convenience wrapper for call sites that hold an optional trace (e.g. pulled
    from ``RequestResponseMetadata.trace``) and don't want to repeat the
    ``None`` check at every stage.
    """
    if trace is not None:
        trace.mark(stage)


class RequestTrace:
    """Nanosecond-resolution per-request stage timer.

    Two durations are logged per stage:

    * ``t``  -- elapsed since the trace was created (request anchor).
    * ``dt`` -- elapsed since the previous mark (the stage's own cost).

    Both are printed in microseconds with 3-decimal (nanosecond) precision,
    which is finer than the millisecond resolution normally available.

    Args:
        request_id: Correlation id, shared across processes.
        scope: Tag identifying the emitting layer/process, e.g. ``"api"``,
            ``"engine"``, ``"ascend"``. Printed on every line.
        t0_ns: Optional anchor timestamp in ns. If omitted, ``now`` is used.
    """

    __slots__ = ("request_id", "scope", "t0_ns", "_last_ns")

    def __init__(
        self,
        request_id: str,
        scope: str = "api",
        *,
        t0_ns: int | None = None,
    ) -> None:
        self.request_id = request_id
        self.scope = scope
        now = time.perf_counter_ns()
        self.t0_ns = now if t0_ns is None else t0_ns
        self._last_ns = self.t0_ns

    def mark(self, stage: str) -> None:
        """Record ``stage`` and log one line at >ms (microsecond) precision.

        A no-op when ``VLLM_TRACE_REQUEST`` is unset/``0``.
        """
        if not trace_enabled():
            return
        now = time.perf_counter_ns()
        t_us = (now - self.t0_ns) / 1_000
        dt_us = (now - self._last_ns) / 1_000
        self._last_ns = now
        _logger.info(
            "[trace %s %s] %-22s t=%11.3f us  dt=%11.3f us",
            self.scope,
            self.request_id,
            stage,
            t_us,
            dt_us,
        )
