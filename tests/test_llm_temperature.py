"""The configurable LLM sampling temperature is forwarded to LiteLLM.

`ingest_temperature:` in config.yaml (or global.yaml) is resolved into a
process-wide stash (see test_config.py) and read at the LiteLLM call sites in
openkb.agent.compiler. These tests pin the call-site behavior: a configured
temperature is forwarded to `litellm.(a)completion`, and nothing is forwarded
when it is unset (so LiteLLM keeps applying the provider's own default).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from openkb.agent.compiler import _llm_call, _llm_call_async
from openkb.config import set_ingest_temperature


def _fake_response():
    choice = MagicMock()
    choice.message.content = "ok"
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_llm_call_forwards_configured_temperature():
    set_ingest_temperature(0.3)
    with patch(
        "openkb.agent.compiler.litellm.completion", return_value=_fake_response()
    ) as completion:
        _llm_call("gpt-4o", [{"role": "user", "content": "hi"}], "step")
    assert completion.call_args.kwargs["temperature"] == 0.3


def test_llm_call_forwards_zero_temperature():
    # 0 is a legitimate (deterministic) temperature — must not be treated as
    # falsy/unset the way `if temperature:` would.
    set_ingest_temperature(0.0)
    with patch(
        "openkb.agent.compiler.litellm.completion", return_value=_fake_response()
    ) as completion:
        _llm_call("gpt-4o", [{"role": "user", "content": "hi"}], "step")
    assert completion.call_args.kwargs["temperature"] == 0.0


def test_llm_call_omits_temperature_when_unset():
    set_ingest_temperature(None)
    with patch(
        "openkb.agent.compiler.litellm.completion", return_value=_fake_response()
    ) as completion:
        _llm_call("gpt-4o", [{"role": "user", "content": "hi"}], "step")
    assert "temperature" not in completion.call_args.kwargs


def test_llm_call_does_not_override_explicit_temperature():
    # An explicit per-call temperature kwarg wins over the configured default.
    set_ingest_temperature(0.9)
    with patch(
        "openkb.agent.compiler.litellm.completion", return_value=_fake_response()
    ) as completion:
        _llm_call("gpt-4o", [{"role": "user", "content": "hi"}], "step", temperature=0.1)
    assert completion.call_args.kwargs["temperature"] == 0.1


def test_llm_call_async_forwards_configured_temperature():
    set_ingest_temperature(0.5)
    with patch(
        "openkb.agent.compiler.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_fake_response(),
    ) as acompletion:
        asyncio.run(_llm_call_async("gpt-4o", [{"role": "user", "content": "hi"}], "step"))
    assert acompletion.call_args.kwargs["temperature"] == 0.5


def test_llm_call_async_omits_temperature_when_unset():
    set_ingest_temperature(None)
    with patch(
        "openkb.agent.compiler.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_fake_response(),
    ) as acompletion:
        asyncio.run(_llm_call_async("gpt-4o", [{"role": "user", "content": "hi"}], "step"))
    assert "temperature" not in acompletion.call_args.kwargs


def test_llm_call_async_does_not_override_explicit_temperature():
    set_ingest_temperature(0.5)
    with patch(
        "openkb.agent.compiler.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_fake_response(),
    ) as acompletion:
        asyncio.run(
            _llm_call_async("gpt-4o", [{"role": "user", "content": "hi"}], "step", temperature=0.1)
        )
    assert acompletion.call_args.kwargs["temperature"] == 0.1
