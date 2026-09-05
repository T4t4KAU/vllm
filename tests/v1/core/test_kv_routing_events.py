# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest
import torch

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine.kv_routing import GPUCacheRoutingIndex
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("internal_dp", [False, True])
def test_dense_dp_scheduler_exports_real_cache_events(
    tmp_path, monkeypatch, uniform, internal_dp
):
    # A local config avoids model or tokenizer downloads in this scheduler test.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "vocab_size": 1000,
            }
        )
    )
    monkeypatch.setenv("VLLM_AGENTRIX_DP_KV_EVENTS", "1")
    monkeypatch.setenv("VLLM_AGENTRIX_DP_ROUTING_POLICY", "session_aware")
    config = VllmConfig(
        model_config=ModelConfig(model=str(tmp_path), skip_tokenizer_init=True),
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=128,
            max_num_batched_tokens=128,
            max_num_seqs=4,
        ),
    )
    config.cache_config.num_gpu_blocks = 16
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    if uniform:
        spec = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"layer": spec})
    cache = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    register_all_kvcache_specs(config)
    scheduler = Scheduler(
        config,
        generate_scheduler_kv_cache_config([cache]),
        StructuredOutputManager(config),
        block_size=16,
        include_finished_set=internal_dp,
    )
    # Dense DP workers have data_parallel_size=1 after core initialization.
    assert config.parallel_config.data_parallel_size == 1
    init_none_hash(sha256)
    request = Request(
        request_id="test",
        prompt_token_ids=list(range(33)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )
    scheduler.add_request(request)
    scheduled = scheduler.schedule()
    outputs = scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=["test"],
            req_id_to_index={"test": 0},
            sampled_token_ids=[[1]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    if not internal_dp:
        assert outputs[0].kv_cache_event_payload is None
        return
    index = GPUCacheRoutingIndex(2, 16, 16)
    assert outputs[0].kv_cache_event_payload is not None
    index.update(0, outputs[0].kv_cache_event_payload)
    assert index.lookup(list(range(33))) == [2, None]
