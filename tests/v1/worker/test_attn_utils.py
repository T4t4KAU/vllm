# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.attention.chunked_local_attention import (
    create_chunked_local_attention_backend,
)
from vllm.v1.attention.backends.fork_attn import ForkAttentionBackend
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheTensor,
    KVQuantMode,
    MambaSpec,
)
from vllm.v1.worker.gpu.attn_utils import (
    _reshape_kv_cache,
    set_fork_attention_cpu_metadata,
    uses_fork_attention_planner,
)
from vllm.v1.worker.utils import AttentionGroup


class FakeFlashAttentionBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3, 4)


class FakeHNDFlashAttentionBackend(FakeFlashAttentionBackend):
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 3, 2, 4)


class RecordingMetadataBuilder:
    def __init__(self) -> None:
        self.cpu_metadata: tuple[object, ...] | None = None

    def set_cpu_metadata(self, *cpu_metadata: object) -> None:
        self.cpu_metadata = cpu_metadata


def test_chunked_local_attention_does_not_receive_global_fork_metadata() -> None:
    full_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float16,
    )
    chunked_spec = ChunkedLocalAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float16,
        attention_chunk_size=32,
    )
    chunked_backend = create_chunked_local_attention_backend(
        ForkAttentionBackend, attention_chunk_size=32
    )
    full_group = AttentionGroup(ForkAttentionBackend, ["full"], full_spec, 0)
    chunked_group = AttentionGroup(chunked_backend, ["chunked"], chunked_spec, 0)
    full_builder = RecordingMetadataBuilder()
    chunked_builder = RecordingMetadataBuilder()
    full_group.metadata_builders = [full_builder]  # type: ignore[list-item]
    chunked_group.metadata_builders = [chunked_builder]  # type: ignore[list-item]

    # The wrapper inherits the underlying name, which is insufficient to decide
    # whether its builder can consume the untransformed global CPU metadata.
    assert chunked_backend.get_name() == "FORK_ATTN"
    assert uses_fork_attention_planner(full_group)
    assert not uses_fork_attention_planner(chunked_group)

    seq_lens = torch.tensor([48], dtype=torch.int32)
    block_table = torch.tensor([[10, 11, 12]], dtype=torch.int32).numpy()
    set_fork_attention_cpu_metadata(
        [[full_group, chunked_group]], seq_lens, [block_table]
    )

    assert full_builder.cpu_metadata is not None
    assert chunked_builder.cpu_metadata is None


def test_reshape_padded_flash_attention_kv_cache_strides_by_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 1, 2)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == spec.real_page_size_bytes // 2 // 4
    assert kv_cache[1, 0].storage_offset() == spec.page_size_bytes // 4
    assert (
        kv_cache[1, 1].storage_offset()
        == (spec.page_size_bytes + spec.real_page_size_bytes // 2) // 4
    )


def test_reshape_padded_hnd_flash_attention_kv_cache_strides_by_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=3,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=1024,
    )
    assert spec.real_page_size_bytes == 768

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeHNDFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 3, 2)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == spec.real_page_size_bytes // 2 // 4
    assert kv_cache.stride(2) == 2
    assert kv_cache.stride(3) == spec.block_size * spec.head_size
    assert kv_cache[1, 0].storage_offset() == spec.page_size_bytes // 4
    assert (
        kv_cache[1, 1].storage_offset()
        == (spec.page_size_bytes + spec.real_page_size_bytes // 2) // 4
    )
    assert (
        kv_cache[1, 1, 3, 2].storage_offset()
        == (
            spec.page_size_bytes
            + spec.real_page_size_bytes // 2
            + 3 * spec.head_size * 4
            + 2 * spec.block_size * spec.head_size * 4
        )
        // 4
    )


class FakeDiffKVBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size * 2)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3)


def test_reshape_padded_diff_kv_cache_does_not_infer_kv_dim():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeDiffKVBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 16, 1, 4)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == 4


class FakePerTokenScaleBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size + 4)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3, 4)


def test_reshape_padded_quantized_kv_cache_preserves_scale_stride():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 128
    assert spec.page_size_bytes == 384

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakePerTokenScaleBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "int8_per_token_head",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 1, 8)
    assert kv_cache.stride(0) == spec.page_size_bytes
    assert kv_cache.stride(1) == 16 * 1 * 8
    assert kv_cache[1, 1].storage_offset() == spec.page_size_bytes + 16 * 1 * 8


class FakeKVFirstBackend:
    """ROCm-style backend that puts K and V ahead of the block dim."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_block_dim(
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        return 1


def _kv_first_setup(shared_by: list[str]):
    num_blocks = 3
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((64,),),
        dtypes=(torch.float32,),
    )
    assert attn_spec.page_size_bytes == mamba_spec.page_size_bytes == 256

    raw_tensor = torch.zeros(attn_spec.page_size_bytes * num_blocks, dtype=torch.int8)
    raw_tensors = {name: raw_tensor for name in shared_by}
    attn_groups = [
        AttentionGroup(
            backend=FakeKVFirstBackend,
            layer_names=["attn"],
            kv_cache_spec=attn_spec,
            kv_cache_group_id=0,
        )
    ]
    if "mamba" in shared_by:
        attn_groups.append(
            AttentionGroup(
                backend=FakeKVFirstBackend,
                layer_names=["mamba"],
                kv_cache_spec=mamba_spec,
                kv_cache_group_id=1,
            )
        )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=raw_tensor.numel(), shared_by=list(shared_by))
        ],
        kv_cache_groups=[],
    )

    kv_caches = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [attn_spec.block_size] * len(attn_groups),
        {},
        kv_cache_config,
    )
    return num_blocks, kv_caches["attn"]


def test_reshape_kv_first_kv_cache_pages_blocks_when_shared_with_mamba():
    num_blocks, kv_cache = _kv_first_setup(["attn", "mamba"])

    assert kv_cache.shape == (2, num_blocks, 16, 1, 2)
    # Block b has to own page b, so its K and V sit side by side within it.
    page = 16 * 1 * 2 * 2
    for block in range(num_blocks):
        assert kv_cache[0, block].storage_offset() == block * page
        assert kv_cache[1, block].storage_offset() == block * page + page // 2


def test_reshape_kv_first_kv_cache_keeps_layout_without_mamba():
    num_blocks, kv_cache = _kv_first_setup(["attn"])

    assert kv_cache.shape == (2, num_blocks, 16, 1, 2)
    # Nothing else indexes this allocation by page, so K and V stay split into
    # one contiguous half each.
    assert kv_cache[1, 0].storage_offset() == num_blocks * 16 * 1 * 2
