# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

from vllm.engine.protocol import EngineClient

router = APIRouter()


class ToolKVTrimRequest(BaseModel):
    """A request to trim one idle resumable session."""

    request_id: str


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/v1/agentrix/tool-kv/trim")
async def trim_tool_kv(
    request: ToolKVTrimRequest, raw_request: Request
) -> dict[str, object]:
    """Release GPU KV blocks held by a streaming session waiting for input."""

    return await engine_client(raw_request).trim_tool_kv(request.request_id)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
