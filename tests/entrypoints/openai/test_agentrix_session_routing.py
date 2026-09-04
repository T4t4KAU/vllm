# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest


def _sampling_params(request: ChatCompletionRequest):
    return request.to_sampling_params(max_tokens=8, default_sampling_params={})


def _beam_search_params(request: ChatCompletionRequest):
    return request.to_beam_search_params(max_tokens=8, default_sampling_params={})


def test_chat_request_infers_agentrix_turn(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_AGENTRIX_DP_ROUTING_POLICY", "session_aware")
    request = ChatCompletionRequest(
        model="model",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "tool call"},
            {"role": "tool", "content": "tool result", "tool_call_id": "call"},
        ],
    )

    assert _sampling_params(request).extra_args["agentrix_turn"] == 1


def test_explicit_agentrix_turn_overrides_inference(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_AGENTRIX_DP_ROUTING_POLICY", "session_aware")
    request = ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "question"}],
        vllm_xargs={
            "agentrix_session_id": "session-a",
            "agentrix_turn": 7,
            "agentrix_history_tokens": 256,
        },
    )

    extra_args = _sampling_params(request).extra_args
    assert extra_args == {
        "agentrix_session_id": "session-a",
        "agentrix_turn": 7,
        "agentrix_history_tokens": 256,
    }


def test_beam_search_preserves_agentrix_session_metadata(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_AGENTRIX_DP_ROUTING_POLICY", "session_aware")
    request = ChatCompletionRequest(
        model="model",
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "tool call"},
            {"role": "tool", "content": "tool result", "tool_call_id": "call"},
        ],
        vllm_xargs={
            "agentrix_session_id": "session-a",
            "agentrix_history_tokens": 256,
        },
        kv_transfer_params={
            "remote_request_id": "remote-request",
            "remote_block_ids": [[1, 2]],
        },
    )

    assert _beam_search_params(request).extra_args == {
        "agentrix_session_id": "session-a",
        "agentrix_history_tokens": 256,
        "agentrix_turn": 1,
        "kv_transfer_params": {
            "remote_request_id": "remote-request",
            "remote_block_ids": [[1, 2]],
        },
    }


def test_native_policy_does_not_add_agentrix_metadata(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_AGENTRIX_DP_ROUTING_POLICY", "native")
    request = ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "question"}],
    )

    assert _sampling_params(request).extra_args is None


def test_completion_beam_search_preserves_vllm_xargs() -> None:
    vllm_xargs = {
        "agentrix_session_id": "session-a",
        "agentrix_turn": 7,
        "agentrix_history_tokens": 256,
        "custom_extension_arg": "custom-value",
    }
    request = CompletionRequest(
        model="model",
        prompt="question",
        use_beam_search=True,
        vllm_xargs=vllm_xargs,
        kv_transfer_params={"do_remote_decode": True},
    )

    sampling_params = request.to_sampling_params(max_tokens=8)
    beam_search_params = request.to_beam_search_params(max_tokens=8)

    assert beam_search_params.extra_args == vllm_xargs
    assert sampling_params.extra_args == {
        **vllm_xargs,
        "kv_transfer_params": {"do_remote_decode": True},
    }
    assert request.vllm_xargs == vllm_xargs
