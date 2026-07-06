# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    parse_fanout_layerwise_load,
    resolve_fanout_layerwise_load,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    _resolve_fanout_chunk_blocks,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("auto", None),
        (None, None),
    ],
)
def test_parse_fanout_layerwise_load(value: object, expected: bool | None) -> None:
    assert parse_fanout_layerwise_load(value) is expected


def test_resolve_fanout_layerwise_load_keeps_full_graph() -> None:
    assert not resolve_fanout_layerwise_load(
        "auto",
        fanout_offload=True,
        has_full_cudagraphs=True,
        estimated_load_bytes=1 << 40,
        threshold_bytes=1,
    )


def test_resolve_fanout_layerwise_load_uses_threshold_without_full_graph() -> None:
    assert resolve_fanout_layerwise_load(
        "auto",
        fanout_offload=True,
        has_full_cudagraphs=False,
        estimated_load_bytes=1024,
        threshold_bytes=512,
    )
    assert not resolve_fanout_layerwise_load(
        "auto",
        fanout_offload=True,
        has_full_cudagraphs=False,
        estimated_load_bytes=256,
        threshold_bytes=512,
    )


def test_parse_fanout_layerwise_load_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="fanout_layerwise_load"):
        parse_fanout_layerwise_load("sometimes")


def test_resolve_fanout_chunk_blocks_defaults_from_tokens() -> None:
    assert (
        _resolve_fanout_chunk_blocks(
            {},
            min_offloaded_block_size=16,
            fanout_budget_blocks=64,
        )
        == 64
    )


def test_resolve_fanout_chunk_blocks_honors_explicit_blocks() -> None:
    assert (
        _resolve_fanout_chunk_blocks(
            {"fanout_chunk_blocks": 128},
            min_offloaded_block_size=16,
            fanout_budget_blocks=256,
        )
        == 128
    )


def test_resolve_fanout_chunk_blocks_rejects_explicit_blocks_over_budget() -> None:
    with pytest.raises(ValueError, match="fanout_chunk_blocks"):
        _resolve_fanout_chunk_blocks(
            {"fanout_chunk_blocks": 128},
            min_offloaded_block_size=16,
            fanout_budget_blocks=64,
        )


def test_resolve_fanout_chunk_blocks_rejects_invalid_tokens() -> None:
    with pytest.raises(ValueError, match="fanout_chunk_tokens"):
        _resolve_fanout_chunk_blocks(
            {"fanout_chunk_tokens": 0},
            min_offloaded_block_size=16,
            fanout_budget_blocks=64,
        )
