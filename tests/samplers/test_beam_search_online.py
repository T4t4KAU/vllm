# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import CompletionOutput, RequestOutput
from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
from vllm.logprobs import Logprob
from vllm.sampling_params import BeamSearchParams


class _Tokenizer:
    eos_token_id = 0

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class _Renderer:
    def get_tokenizer(self) -> _Tokenizer:
        return _Tokenizer()


class _EngineClient:
    def __init__(self) -> None:
        self.extra_args_by_prompt: list[tuple[list[int], dict]] = []

    async def generate(self, prompt, *args, **kwargs):
        self.extra_args_by_prompt.append(
            (list(prompt["prompt_token_ids"]), dict(args[0].extra_args or {}))
        )
        yield RequestOutput(
            request_id=kwargs.get("request_id", "test-request"),
            prompt=prompt.get("prompt"),
            prompt_token_ids=prompt["prompt_token_ids"],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=[],
                    cumulative_logprob=None,
                    logprobs=[
                        {
                            11: Logprob(logprob=-1.0),
                            12: Logprob(logprob=-2.0),
                            13: Logprob(logprob=-3.0),
                            14: Logprob(logprob=-4.0),
                            _Tokenizer.eos_token_id: Logprob(logprob=-0.1),
                        }
                    ],
                    finish_reason=None,
                )
            ],
            finished=True,
        )


class _Serving(BeamSearchOnlineMixin):
    renderer = _Renderer()

    def __init__(self) -> None:
        self.engine_client = _EngineClient()


@pytest.mark.asyncio
async def test_beam_search_handles_extra_logprob_candidates() -> None:
    prompt = {
        "type": "token",
        "prompt": "prompt",
        "prompt_token_ids": [1],
    }
    params = BeamSearchParams(beam_width=2, max_tokens=1)

    outputs = [
        output async for output in _Serving().beam_search(prompt, "request", params)
    ]

    assert len(outputs) == 1
    assert outputs[0].outputs[0].finish_reason == "stop"
    assert outputs[0].outputs[0].token_ids == []
    assert outputs[0].outputs[0].cumulative_logprob == pytest.approx(-0.1)


@pytest.mark.asyncio
async def test_beam_search_tracks_internal_steps_without_changing_turn() -> None:
    prompt = {
        "type": "token",
        "prompt": "prompt",
        "prompt_token_ids": [1],
    }
    extra_args = {
        "agentrix_session_id": "session-a",
        "agentrix_turn": 0,
        "agentrix_history_tokens": 128,
        "kv_transfer_params": {
            "remote_request_id": "remote-request",
            "remote_block_ids": [[1, 2]],
        },
    }
    params = BeamSearchParams(
        beam_width=2,
        max_tokens=2,
        extra_args=extra_args,
    )
    serving = _Serving()

    _ = [output async for output in serving.beam_search(prompt, "request", params)]

    requests = serving.engine_client.extra_args_by_prompt
    assert len(requests) == 3
    assert [len(prompt_token_ids) for prompt_token_ids, _ in requests] == [1, 2, 2]
    assert [args for _, args in requests] == [
        {
            "agentrix_session_id": "session-a",
            "agentrix_turn": 0,
            "agentrix_history_tokens": 128,
            "kv_transfer_params": {
                "remote_request_id": "remote-request",
                "remote_block_ids": [[1, 2]],
            },
            "agentrix_beam_step": 0,
        },
        {
            "agentrix_session_id": "session-a",
            "agentrix_turn": 0,
            "agentrix_history_tokens": 128,
            "agentrix_beam_step": 1,
        },
        {
            "agentrix_session_id": "session-a",
            "agentrix_turn": 0,
            "agentrix_history_tokens": 128,
            "agentrix_beam_step": 1,
        },
    ]
    assert extra_args == {
        "agentrix_session_id": "session-a",
        "agentrix_turn": 0,
        "agentrix_history_tokens": 128,
        "kv_transfer_params": {
            "remote_request_id": "remote-request",
            "remote_block_ids": [[1, 2]],
        },
    }


@pytest.mark.asyncio
async def test_beam_search_tracks_internal_steps_without_session_turn() -> None:
    prompt = {
        "type": "token",
        "prompt": "prompt",
        "prompt_token_ids": [1],
    }
    params = BeamSearchParams(
        beam_width=2,
        max_tokens=2,
        extra_args={"agentrix_session_id": "session-a"},
    )
    serving = _Serving()

    _ = [output async for output in serving.beam_search(prompt, "request", params)]

    requests = serving.engine_client.extra_args_by_prompt
    assert [args["agentrix_beam_step"] for _, args in requests] == [0, 1, 1]
    assert all(args["agentrix_session_id"] == "session-a" for _, args in requests)


@pytest.mark.asyncio
async def test_beam_search_rejects_output_side_kv_transfer() -> None:
    prompt = {
        "type": "token",
        "prompt": "prompt",
        "prompt_token_ids": [1],
    }
    params = BeamSearchParams(
        beam_width=2,
        max_tokens=2,
        extra_args={"kv_transfer_params": {"do_remote_decode": True}},
    )
    serving = _Serving()

    with pytest.raises(ValueError, match="output-side KV transfer"):
        _ = [output async for output in serving.beam_search(prompt, "request", params)]

    assert not serving.engine_client.extra_args_by_prompt
