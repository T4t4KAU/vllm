# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import msgspec
import pytest

from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, BlockStored
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.kv_routing import GPUCacheRoutingIndex
from vllm.v1.engine.prefix_router import PrefixAwareDPRouter


def _store(
    hashes: list[int], tokens: list[int], parent: int | None = None, **kwargs
) -> BlockStored:
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=tokens,
        block_size=4,
        lora_id=None,
        lora_name=None,
        medium="GPU",
        **kwargs,
    )


def _payload(*events) -> bytes:
    return msgspec.msgpack.encode(list(events))


def test_removed_ancestor_limits_surviving_descendant_prefix() -> None:
    index = GPUCacheRoutingIndex(2, 4, 16)
    tokens = list(range(12))
    assert index.lookup(tokens) == [None, None]
    index.update(0, _payload(_store([1, 2, 3], tokens)))
    index.update(1, _payload())
    assert index.lookup(tokens) == [3, 0]

    index.update(0, _payload(BlockRemoved([2], "GPU")))
    assert index.lookup(tokens) == [1, 0]
    index.update(0, _payload(_store([2], tokens[4:8], parent=1)))
    assert index.lookup(tokens) == [3, 0]
    index.update(0, _payload(AllBlocksCleared()))
    assert index.lookup(tokens) == [0, 0]


def test_capacity_eviction_is_conservative() -> None:
    index = GPUCacheRoutingIndex(2, 4, 2)
    index.update(0, _payload(_store([1, 2, 3], list(range(12)))))
    # The missing root prevents advertising surviving cache descendants.
    assert index.lookup(list(range(12))) == [0, None]


def test_unknown_parents_and_salted_stores_are_not_advertised() -> None:
    index = GPUCacheRoutingIndex(2, 4, 16)
    index.update(0, _payload(_store([1], [1, 2, 3, 4], parent=100)))
    index.update(0, _payload(_store([2], [1, 2, 3, 4], extra_keys=[("salt",)])))
    assert index.lookup([1, 2, 3, 4]) == [0, None]


def test_hybrid_groups_disable_physical_hint_for_that_rank() -> None:
    index = GPUCacheRoutingIndex(2, 4, 16)
    index.update(0, _payload(_store([1], [1, 2, 3, 4])))
    index.update(0, _payload(_store([2], [1, 2, 3, 4], group_idx=1)))
    assert index.lookup([1, 2, 3, 4]) == [None, None]


def test_cache_event_payload_roundtrips_through_engine_output() -> None:
    original = EngineCoreOutputs(
        engine_index=1, kv_cache_event_payload=_payload(_store([1], [1, 2, 3, 4]))
    )
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(original), type=EngineCoreOutputs
    )
    index = GPUCacheRoutingIndex(2, 4, 16)
    index.update(decoded.engine_index, decoded.kv_cache_event_payload)
    assert index.lookup([1, 2, 3, 4]) == [None, 1]


@pytest.mark.parametrize("overloaded", [False, True])
def test_session_moves_to_verified_gpu_prefix_with_load_guard(overloaded: bool) -> None:
    router = PrefixAwareDPRouter(
        2, 4, 4, 30, 1, routing_policy="session_aware", use_kv_events=True
    )
    request = SimpleNamespace(
        request_id="first",
        prompt_token_ids=list(range(16)),
        prompt_embeds=None,
        prompt_is_token_ids=None,
        lora_request=None,
        mm_features=None,
        cache_salt=None,
        sampling_params=SimpleNamespace(
            max_tokens=1,
            extra_args={
                "agentrix_session_id": "session",
                "agentrix_turn": 1,
            },
        ),
    )
    router.add_request(request, rank=0)
    router.observe_cache_events(0, _payload())
    router.observe_cache_events(1, _payload(_store([1, 2, 3, 4], list(range(16)))))
    request.request_id = "next"
    counts = [[0, 0], [20, 0] if overloaded else [0, 0]]

    assert router.choose_rank(request, counts, baseline_rank=0) == (
        0 if overloaded else 1
    )
    assert router.cache_event_route_count == 1
