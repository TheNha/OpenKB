"""Process-wide LLM request runtime settings (CLI path).

Holds the extra_headers / extra_body / timeout / parallel_tool_calls state the
CLI resolves once from a KB's config (``cli._setup_llm_key``) and every
tool-using agent builder reads back via :func:`resolve_model_settings`. Split
out of ``openkb/config.py`` (see ``docs/golden-principles.md#file-size``) —
this block has no dependency on the rest of that module. The REST API path
does NOT use these globals (see :class:`openkb.config.LlmCredentialBundle`);
they exist so concurrent requests on a shared event-loop thread never see each
other's key/headers/timeout.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger("openkb.config")

_runtime_extra_headers: dict[str, str] = {}


def set_extra_headers(headers: dict[str, str]) -> None:
    """Set the process-wide extra headers for LLM requests."""
    global _runtime_extra_headers
    _runtime_extra_headers = dict(headers)


def get_extra_headers() -> dict[str, str]:
    """Return a copy of the process-wide extra headers for LLM requests."""
    return dict(_runtime_extra_headers)


_runtime_extra_body: dict[str, Any] = {}


def set_extra_body(extra_body: dict[str, Any]) -> None:
    """Set the process-wide extra request-body fields for LLM requests."""
    global _runtime_extra_body
    _runtime_extra_body = dict(extra_body)


def get_extra_body() -> dict[str, Any]:
    """Return a copy of the process-wide extra request-body fields for LLM requests."""
    return dict(_runtime_extra_body)


# Process-wide LLM request timeout (seconds), set from config by the CLI and
# read at the call sites via get_timeout(). None = use LiteLLM's default.
_runtime_timeout: float | None = None


def set_timeout(timeout: float | None) -> None:
    """Set the process-wide LLM request timeout in seconds; ``None`` clears it."""
    global _runtime_timeout
    _runtime_timeout = timeout


def get_timeout() -> float | None:
    """Return the process-wide LLM request timeout in seconds, or ``None``."""
    return _runtime_timeout


def get_timeout_extra_args() -> dict[str, float] | None:
    """Timeout as Agents-SDK ``ModelSettings.extra_args`` (it has no ``timeout``
    field), or ``None``. The LiteLLM provider forwards it to the completion call.
    """
    return {"timeout": _runtime_timeout} if _runtime_timeout is not None else None


# Process-wide LLM sampling temperature, split by phase — ingest (compiler.py
# + PageIndex) and query/chat (query.py) are separate call sites a user may
# want tuned differently (e.g. low temperature for consistent wiki writing,
# higher for conversational answers). Set from config by the CLI, read at the
# call sites via get_ingest_temperature()/get_query_temperature(). None = use
# the provider's default.
_runtime_ingest_temperature: float | None = None
_runtime_query_temperature: float | None = None


def set_ingest_temperature(temperature: float | None) -> None:
    """Set the process-wide LLM sampling temperature for ingest (compile +
    PageIndex) calls; ``None`` clears it."""
    global _runtime_ingest_temperature
    _runtime_ingest_temperature = temperature


def get_ingest_temperature() -> float | None:
    """Return the process-wide ingest-phase LLM sampling temperature, or ``None``."""
    return _runtime_ingest_temperature


def set_query_temperature(temperature: float | None) -> None:
    """Set the process-wide LLM sampling temperature for query/chat calls;
    ``None`` clears it."""
    global _runtime_query_temperature
    _runtime_query_temperature = temperature


def resolve_temperature(config: dict, key: str = "temperature") -> float | None:
    """Resolve an optional temperature key to a finite, non-negative float.

    ``key`` lets ingest and query use distinct config keys
    (``ingest_temperature`` / ``query_temperature``) sharing one validator.
    Mirrors ``openkb.config.resolve_timeout``, except ``0`` is valid here (a
    deterministic temperature), so only bools/non-numeric/nan/inf/negative
    are rejected.
    """
    raw = config.get(key)
    if raw is None:
        return None
    value = None if isinstance(raw, bool) else raw
    try:
        value = float(value) if isinstance(value, (int, float, str)) else None
    except (TypeError, ValueError):
        value = None
    if value is None or not math.isfinite(value) or value < 0:
        logger.warning(
            "config: '%s' must be a finite non-negative number, got %r — ignoring it.",
            key,
            raw,
        )
        return None
    return value


def get_query_temperature() -> float | None:
    """Return the process-wide query/chat-phase LLM sampling temperature, or ``None``."""
    return _runtime_query_temperature


# Process-wide agent ``parallel_tool_calls`` as ``(value, was_explicit)``, set
# from config by the CLI and read when building agents. ``(None, False)`` = not
# configured, so each agent falls back to its own default (resolve_model_settings).
_runtime_parallel_tool_calls: tuple[bool | None, bool] = (None, False)


def set_parallel_tool_calls(value: bool | None, was_explicit: bool) -> None:
    """Set the process-wide ``parallel_tool_calls`` — see
    :func:`openkb.config.resolve_parallel_tool_calls`.
    """
    global _runtime_parallel_tool_calls
    _runtime_parallel_tool_calls = (value, was_explicit)


def get_parallel_tool_calls() -> tuple[bool | None, bool]:
    """Return the process-wide ``parallel_tool_calls`` as ``(value, was_explicit)``."""
    return _runtime_parallel_tool_calls


def resolve_model_settings(*, default_parallel_tool_calls: bool | None = False) -> dict[str, Any]:
    """Assemble the agents-SDK ``ModelSettings`` kwargs from the process-wide LLM
    runtime settings — the single place tool-using agent builders wire them in.

    ``default_parallel_tool_calls`` (the caller's own historical default) is used
    only when config didn't set ``parallel_tool_calls``; an explicit value always
    wins. Tool-less agents (skill-eval graders) skip this and omit the setting —
    the SDK forwards an explicit ``False`` even without tools, which strict
    OpenAI-compatible endpoints reject.
    """
    value, was_explicit = get_parallel_tool_calls()
    parallel_tool_calls = value if was_explicit else default_parallel_tool_calls
    return {
        "extra_headers": get_extra_headers() or None,
        "extra_body": get_extra_body() or None,
        "extra_args": get_timeout_extra_args(),
        "parallel_tool_calls": parallel_tool_calls,
        "temperature": get_query_temperature(),
    }
