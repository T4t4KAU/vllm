# Prefix and session affinity for internal DP

Agentrix adds optional affinity to v0.28.0's internal data-parallel load
balancer. The default remains `native`. Attention selection is independent:
the router works with FlashAttention and ForkAttention, and does not change
Fork's 16,384-token, eight-query admission policy.

```bash
VLLM_AGENTRIX_DP_ROUTING_POLICY=prefix_aware \
vllm serve <model> \
    --data-parallel-size 2 \
    --api-server-count 1 \
    --enable-prefix-caching
```

Choose one policy with `VLLM_AGENTRIX_DP_ROUTING_POLICY`:

| Policy | Behavior |
| --- | --- |
| `native` | Unmodified upstream load selection. |
| `prefix_aware` | Prefer a known prefix within load and estimated-work limits. |
| `session_aware` | Prefer the session owner when its prefix and load qualify; otherwise use bounded prefix affinity or native selection. |
| `session_sticky` | Place a new explicit session using native selection; preserve its owner until expiry or reset. Without a session ID, use bounded prefix affinity. |

Send the upstream `session_id` request field or `X-Session-ID` header for
multi-turn affinity. Upstream also accepts `vllm_xargs.session_id`.
`vllm_xargs.agentrix_session_id` remains a legacy alias; the upstream field
takes precedence. A legacy `agentrix_turn: 0` explicitly starts a new turn-zero
placement under session policies. Optional `agentrix_history_tokens` constrains
the session-owner preference under `session_aware`. Beam steps after zero use
prefix routing without changing their parent session's mapping.

## Integration

The router consumes the upstream load score, including the exact local
in-flight count, coordinator queue counts and KV-pressure penalty. Native
selection still determines cold placement and tie rotation. Explicit
`X-data-parallel-rank` and late-interaction pooling placement take precedence.
Completions and aborts use the existing engine ownership bookkeeping.

Bounded affinity allows a load score within four of the lightest replica and
estimated work within 8,192 token-equivalent units of the cheapest eligible
replica. Work includes uncached prompt tokens and maximum output length weighted
by 16; it is a placement estimate, not measured GPU execution time.
`session_aware` additionally limits eligible load to twice the mean, with a floor
of one. Its session-owner preference requires at least half the supplied history
when that hint exists; bounded prefix affinity remains available otherwise.
Strict `session_sticky` deliberately preserves ownership independently of load.

Prefix hints use chained full-block token hashes, separated by cache salt,
LoRA identity and multimodal identity. A first generated output establishes a
completed-prompt hint; queued requests and scheduling events alone do not.
Streaming extends the chain only through computed tokens, excluding the last
sampled token until the next model output. Preemption events invalidate active
hints. Cache reset and sleep invalidate all hints, including late outputs from
requests already in flight.

Warm hints and session mappings expire after 300 seconds. Each has a capacity
of 1,024 requests/sessions; warm prefix checkpoints are additionally capped at
262,144, with at most 256 checkpoints per request. Hints do not pin GPU blocks
or guarantee that APC still owns them. The destination engine performs its
normal cache lookup and allocation.

## Supported deployment

Affinity requires internal DP with one API frontend, fixed ranks, and prefix
caching enabled. Multiple API frontends and elastic EP are rejected when
affinity is explicitly selected, since they do not share this routing state.
Resumable requests, pooling, prompt-logprob requests, prompt embeddings without
a reproducible cache identity, and requests that skip prefix-cache reads use
native routing.
Both synchronous and asynchronous scheduling can use this frontend policy;
ForkAttention retains its own scheduling requirements.

## Validation

The existing `tests/v1/engine/test_engine_core_client.py` suite covers prefix
reuse, session ownership, cache namespace isolation, explicit placement,
capacity/TTL/reset, preemption, native burst balancing, KV-pressure handling,
and completion accounting. Model-output and official AgentX comparisons use
two DP replicas with TP=1 and the same GPU/cache budget for each policy.
