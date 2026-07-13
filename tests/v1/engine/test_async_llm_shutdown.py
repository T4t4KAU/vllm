# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import vllm.v1.engine.async_llm as async_llm_module
from vllm.v1.engine.async_llm import AsyncLLM


def test_shutdown_cancels_output_handler_before_engine_core(monkeypatch):
    events: list[str] = []
    handler = object()
    renderer = MagicMock()
    renderer.shutdown.side_effect = lambda: events.append("renderer")
    engine_core = MagicMock()
    engine_core.shutdown.side_effect = lambda timeout: events.append("engine_core")
    engine = SimpleNamespace(
        renderer=renderer,
        output_handler=handler,
        engine_core=engine_core,
    )

    monkeypatch.setattr(async_llm_module, "shutdown_prometheus", lambda: None)
    monkeypatch.setattr(
        async_llm_module,
        "cancel_task_threadsafe",
        lambda task: events.append("output_handler"),
    )

    AsyncLLM.shutdown(engine, timeout=1.0)

    assert events == ["renderer", "output_handler", "engine_core"]
    engine_core.shutdown.assert_called_once_with(timeout=1.0)
