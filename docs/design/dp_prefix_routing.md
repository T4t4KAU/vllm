# Official data-parallel routing

Agentrix uses the published [vLLM router](https://github.com/vllm-project/router)
instead of a local prefix/session policy inside `DPLBAsyncMPClient`.
The old `VLLM_AGENTRIX_DP_ROUTING_POLICY` option has been removed.

Use `consistent_hash` with `X-Session-ID` for multi-turn affinity, or select
`cache_aware` using the official router CLI. Enable DP rank forwarding with
`--intra-node-data-parallel-size`. Requests without explicit rank placement
continue through upstream internal load balancing when sent directly to vLLM.
Attention backend selection remains independent of routing.

In the enclosing Agentrix repository, `benchmark/requirements-router.txt` pins
the published package and `benchmark/scripts/serve_dp_router.py` checks that
version before invoking its official Rust implementation. See
`docs/dp_routing.md` in that repository for deployment and historical results.

Cache resets, engine metrics and explicitly pinned diagnostic requests go to
the backend control endpoint. Router cache estimates are not proof of physical
KV residency. The client retains one correctness fix: a DP cache reset returns
success only when every engine reports success.
