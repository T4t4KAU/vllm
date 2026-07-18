# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.tool_kv.api_router import attach_router


def test_trim_tool_kv_endpoint() -> None:
    app = FastAPI()
    engine_client = MagicMock()
    engine_client.trim_tool_kv = AsyncMock(
        return_value={
            "request_id": "session",
            "trimmed": True,
            "kv_cache_usage_before": 0.5,
            "kv_cache_usage_after": 0.0,
        }
    )
    app.state.engine_client = engine_client
    attach_router(app)

    response = TestClient(app).post(
        "/v1/agentrix/tool-kv/trim", json={"request_id": "session"}
    )

    assert response.status_code == 200
    assert response.json()["trimmed"] is True
    engine_client.trim_tool_kv.assert_awaited_once_with("session")
