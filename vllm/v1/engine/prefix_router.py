# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest

PrefixKey = tuple[int, int]


@dataclass
class _RequestPrefix:
    rank: int
    namespace: int
    chain: int
    keys: list[PrefixKey]
    tail: list[int]
    work_units: int
    active: bool = False


@dataclass
class _EngineTelemetry:
    graph_kind: str = ""
    graph_capacity: int = 0
    active_ctas: int = 0
    shared_ctas: int = 0
    singleton_ctas: int = 0
    kv_cache_usage: float = 0.0


class PrefixAwareDPRouter:
    """Request-local logical prefix state for internal DP routing.

    The router deliberately treats logical prefix matches as a placement hint.
    Physical KV reuse and correctness remain owned by each engine rank.
    """

    def __init__(
        self,
        num_ranks: int,
        block_size: int,
        load_slack: int,
        warm_ttl_s: float,
        min_prefix_blocks: int,
        *,
        max_warm_requests: int = 1024,
        graph_buckets: Sequence[int] = (),
        graph_slack_buckets: int = 1,
        prefix_chunk_blocks: int = 128,
        work_slack_tokens: int = 8192,
        decode_token_weight: int = 16,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if num_ranks <= 1:
            raise ValueError("prefix-aware DP routing requires multiple ranks")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if load_slack < 0:
            raise ValueError("load_slack must be non-negative")
        if warm_ttl_s < 0:
            raise ValueError("warm_ttl_s must be non-negative")
        if min_prefix_blocks <= 0:
            raise ValueError("min_prefix_blocks must be positive")
        if max_warm_requests < 0:
            raise ValueError("max_warm_requests must be non-negative")
        if graph_slack_buckets < 0:
            raise ValueError("graph_slack_buckets must be non-negative")
        if prefix_chunk_blocks <= 0:
            raise ValueError("prefix_chunk_blocks must be positive")
        if work_slack_tokens < 0:
            raise ValueError("work_slack_tokens must be non-negative")
        if decode_token_weight <= 0:
            raise ValueError("decode_token_weight must be positive")

        self.num_ranks = num_ranks
        self.block_size = block_size
        self.load_slack = load_slack
        self.warm_ttl_s = warm_ttl_s
        self.min_prefix_blocks = min_prefix_blocks
        self.max_warm_requests = max_warm_requests
        self.graph_buckets = tuple(sorted(set(graph_buckets)))
        self.graph_slack_buckets = graph_slack_buckets
        self.prefix_chunk_blocks = prefix_chunk_blocks
        self.work_slack_tokens = work_slack_tokens
        self.decode_token_weight = decode_token_weight
        self._clock = clock
        self._engine_telemetry = [_EngineTelemetry() for _ in range(num_ranks)]

        self._requests: dict[str, _RequestPrefix] = {}
        self._pending_prefixes: dict[
            str, tuple[int, int, list[PrefixKey], list[int]]
        ] = {}
        self._pending_work: dict[str, int] = {}
        self._rank_work = [0] * num_ranks
        self._live = [Counter[PrefixKey]() for _ in range(num_ranks)]
        self._active = [Counter[PrefixKey]() for _ in range(num_ranks)]
        self._warm = [Counter[PrefixKey]() for _ in range(num_ranks)]
        self._warm_requests: deque[tuple[float, int, list[PrefixKey]]] = deque()

        self.route_count = 0
        self.affinity_route_count = 0
        self.rank_route_counts = [0] * num_ranks
        self.routing_time_ns = 0
        self.last_affinity_blocks = 0
        self.graph_bound_route_count = 0
        self.arrival_wave_count = 0

    def order_arrival_wave(
        self,
        requests: Sequence[EngineCoreRequest],
    ) -> list[int]:
        """Order a pending wave by deepest shared subtree and split regret."""
        prefixes = [self._build_prefix(request) for request in requests]
        key_fanout: Counter[PrefixKey] = Counter(
            key for prefix in prefixes if prefix is not None for key in prefix[2]
        )
        groups: dict[PrefixKey | tuple[str, int], list[int]] = {}
        group_priority: dict[PrefixKey | tuple[str, int], tuple[int, int]] = {}
        for index, (request, prefix) in enumerate(zip(requests, prefixes)):
            if prefix is None:
                anchor: PrefixKey | tuple[str, int] = ("private", index)
                priority = (0, 0)
            else:
                self._pending_prefixes[request.request_id] = prefix
                shared = [key for key in prefix[2] if key_fanout[key] > 1]
                if shared:
                    anchor = shared[-1]
                    fanout = key_fanout[anchor]
                    priority = (anchor[0] * (fanout - 1), fanout)
                else:
                    anchor = ("private", index)
                    priority = (0, 0)
            groups.setdefault(anchor, []).append(index)
            group_priority[anchor] = priority

        self.arrival_wave_count += 1
        ordered_groups = sorted(
            groups,
            key=lambda anchor: (
                -group_priority[anchor][0],
                -group_priority[anchor][1],
                groups[anchor][0],
            ),
        )
        return [index for anchor in ordered_groups for index in groups[anchor]]

    def update_engine_telemetry(
        self,
        telemetry: Sequence[Sequence[object]],
    ) -> None:
        for rank, item in enumerate(telemetry[: self.num_ranks]):
            if len(item) < 2:
                continue
            execution_stats, kv_cache_usage = item
            self.update_rank_telemetry(rank, execution_stats, float(kv_cache_usage))

    def update_rank_telemetry(
        self,
        rank: int,
        execution_stats: object,
        kv_cache_usage: float,
    ) -> None:
        if not 0 <= rank < self.num_ranks:
            return
        state = self._engine_telemetry[rank]
        state.kv_cache_usage = kv_cache_usage
        if not isinstance(execution_stats, (list, tuple)) or len(execution_stats) < 5:
            return
        kind, capacity, active, shared, singleton = execution_stats
        state.graph_kind = str(kind)
        state.graph_capacity = int(capacity)
        state.active_ctas = int(active)
        state.shared_ctas = int(shared)
        state.singleton_ctas = int(singleton)

    def _graph_bucket_index(self, required_ctas: int) -> int:
        for index, bucket in enumerate(self.graph_buckets):
            if required_ctas <= bucket:
                return index
        return len(self.graph_buckets)

    def _predicted_graph_bucket_indices(
        self,
        keys: Sequence[PrefixKey],
    ) -> list[int] | None:
        if not self.graph_buckets:
            return None
        indices: list[int] = []
        for rank, state in enumerate(self._engine_telemetry):
            logical_ctas = set(self._logical_cta_keys(keys))
            for record in self._requests.values():
                if record.rank == rank:
                    logical_ctas.update(self._logical_cta_keys(record.keys))
            predicted_ctas = max(
                state.active_ctas,
                len(logical_ctas),
            )
            indices.append(self._graph_bucket_index(predicted_ctas))
        return indices

    def _logical_cta_keys(self, keys: Sequence[PrefixKey]) -> list[PrefixKey]:
        """Approximate forest CTAs by unique chunk-ending prefix hashes."""
        if not keys:
            return []
        depths = list(
            range(self.prefix_chunk_blocks, len(keys), self.prefix_chunk_blocks)
        )
        depths.append(len(keys))
        return [keys[depth - 1] for depth in depths]

    def _candidate_work(
        self,
        request: EngineCoreRequest,
        keys: Sequence[PrefixKey],
        rank: int,
    ) -> int:
        active_depth, _ = self._deepest_match(keys, self._active[rank])
        live_depth, _ = self._deepest_match(keys, self._live[rank])
        warm_depth, _ = self._deepest_match(keys, self._warm[rank])
        cached_tokens = max(active_depth, live_depth, warm_depth) * self.block_size
        prompt_tokens = len(request.prompt_token_ids or ())
        private_prompt_tokens = max(0, prompt_tokens - cached_tokens)
        sampling_params = getattr(request, "sampling_params", None)
        max_tokens = int(getattr(sampling_params, "max_tokens", 0) or 0)
        return private_prompt_tokens + max_tokens * self.decode_token_weight

    @staticmethod
    def _namespace(request: EngineCoreRequest) -> int | None:
        # Prompt embeds do not carry a stable identity at this layer. Routing
        # them by placeholder token IDs could create false affinity.
        if request.prompt_embeds is not None:
            return None
        if request.prompt_is_token_ids is not None and not all(
            request.prompt_is_token_ids
        ):
            return None

        lora_id = (
            request.lora_request.lora_int_id
            if request.lora_request is not None
            else None
        )
        mm_identity = tuple(
            (
                feature.modality,
                feature.identifier,
                feature.mm_position.offset,
                feature.mm_position.length,
            )
            for feature in (request.mm_features or ())
        )
        return hash((request.cache_salt, lora_id, mm_identity))

    def _build_prefix(
        self, request: EngineCoreRequest
    ) -> tuple[int, int, list[PrefixKey], list[int]] | None:
        token_ids = request.prompt_token_ids
        namespace = self._namespace(request)
        if token_ids is None or namespace is None:
            return None

        chain = namespace
        keys: list[PrefixKey] = []
        complete_tokens = len(token_ids) - len(token_ids) % self.block_size
        for start in range(0, complete_tokens, self.block_size):
            block = tuple(token_ids[start : start + self.block_size])
            chain = hash((chain, block))
            keys.append((len(keys) + 1, chain))
        tail = list(token_ids[complete_tokens:])
        return namespace, chain, keys, tail

    @staticmethod
    def _increment(counter: Counter[PrefixKey], keys: Sequence[PrefixKey]) -> None:
        counter.update(keys)

    @staticmethod
    def _decrement(counter: Counter[PrefixKey], keys: Sequence[PrefixKey]) -> None:
        for key in keys:
            count = counter[key]
            if count <= 1:
                counter.pop(key, None)
            else:
                counter[key] = count - 1

    def _drop_oldest_warm(self) -> None:
        _, rank, keys = self._warm_requests.popleft()
        self._decrement(self._warm[rank], keys)

    def _expire_warm(self) -> None:
        now = self._clock()
        while self._warm_requests and self._warm_requests[0][0] <= now:
            self._drop_oldest_warm()

    @staticmethod
    def _deepest_match(
        keys: Sequence[PrefixKey], counts: Counter[PrefixKey]
    ) -> tuple[int, int]:
        for depth in range(len(keys), 0, -1):
            fanout = counts.get(keys[depth - 1], 0)
            if fanout:
                return depth, fanout
        return 0, 0

    def choose_rank(
        self,
        request: EngineCoreRequest,
        engine_counts: Sequence[Sequence[int]],
        start_index: int = 0,
    ) -> int:
        started_ns = time.perf_counter_ns()
        self._expire_warm()
        prefix = self._pending_prefixes.get(request.request_id)
        if prefix is None:
            prefix = self._build_prefix(request)
        if prefix is not None:
            self._pending_prefixes[request.request_id] = prefix
        load_scores = [waiting * 4 + running for waiting, running in engine_counts]
        min_load = min(load_scores)
        eligible = {
            rank
            for rank, load in enumerate(load_scores)
            if load <= min_load + self.load_slack
        }

        baseline_rank = min(
            range(self.num_ranks),
            key=lambda offset: (
                load_scores[(start_index + offset) % self.num_ranks],
                offset,
            ),
        )
        baseline_rank = (start_index + baseline_rank) % self.num_ranks

        chosen_rank = baseline_rank
        affinity_blocks = 0
        if prefix is not None:
            keys = prefix[2]
            graph_indices = self._predicted_graph_bucket_indices(keys)
            if graph_indices is not None:
                min_graph_index = min(graph_indices)
                max_usage = max(
                    state.kv_cache_usage for state in self._engine_telemetry
                )
                slack = 0 if max_usage >= 0.90 else self.graph_slack_buckets
                graph_eligible = {
                    rank
                    for rank, index in enumerate(graph_indices)
                    if index <= min_graph_index + slack
                }
                if len(graph_eligible) < self.num_ranks:
                    self.graph_bound_route_count += 1
                eligible = (eligible & graph_eligible) or graph_eligible
                if chosen_rank not in eligible:
                    chosen_rank = min(
                        eligible,
                        key=lambda rank: (
                            load_scores[rank],
                            (rank - start_index) % self.num_ranks,
                        ),
                    )
            candidate_work = [
                self._rank_work[rank] + self._candidate_work(request, keys, rank)
                for rank in range(self.num_ranks)
            ]
            min_work = min(candidate_work[rank] for rank in eligible)
            work_eligible = {
                rank
                for rank in eligible
                if candidate_work[rank] <= min_work + self.work_slack_tokens
            }
            eligible = work_eligible or eligible
            scores: list[tuple[tuple[int, ...], int]] = []
            for rank in eligible:
                active_depth, active_fanout = self._deepest_match(
                    keys, self._active[rank]
                )
                live_depth, live_fanout = self._deepest_match(keys, self._live[rank])
                warm_depth, warm_fanout = self._deepest_match(keys, self._warm[rank])
                score = (
                    active_depth,
                    active_fanout,
                    live_depth,
                    live_fanout,
                    warm_depth,
                    warm_fanout,
                    -load_scores[rank],
                    -((rank - start_index) % self.num_ranks),
                )
                scores.append((score, rank))
            best_score, best_rank = max(scores)
            affinity_blocks = max(best_score[0], best_score[2], best_score[4])
            if affinity_blocks >= self.min_prefix_blocks:
                chosen_rank = best_rank
                self.affinity_route_count += 1
            self._pending_work[request.request_id] = self._candidate_work(
                request, keys, chosen_rank
            )

        self.route_count += 1
        self.rank_route_counts[chosen_rank] += 1
        self.last_affinity_blocks = affinity_blocks
        self.routing_time_ns += time.perf_counter_ns() - started_ns
        return chosen_rank

    def add_request(self, request: EngineCoreRequest, rank: int) -> None:
        prefix = self._pending_prefixes.pop(request.request_id, None)
        if prefix is None:
            prefix = self._build_prefix(request)
        if prefix is None:
            return
        namespace, chain, keys, tail = prefix
        if request.request_id in self._requests:
            self.finish_request(request.request_id, keep_warm=False)
        work_units = self._pending_work.pop(
            request.request_id,
            self._candidate_work(request, keys, rank),
        )
        record = _RequestPrefix(rank, namespace, chain, keys, tail, work_units)
        self._requests[request.request_id] = record
        self._rank_work[rank] += work_units
        self._increment(self._live[rank], keys)

    def discard_pending(self, request_ids: Sequence[str]) -> None:
        for request_id in request_ids:
            self._pending_prefixes.pop(request_id, None)
            self._pending_work.pop(request_id, None)

    def _append_tokens(self, record: _RequestPrefix, token_ids: Sequence[int]) -> None:
        if not token_ids:
            return
        record.tail.extend(token_ids)
        while len(record.tail) >= self.block_size:
            block = tuple(record.tail[: self.block_size])
            del record.tail[: self.block_size]
            record.chain = hash((record.chain, block))
            key = (len(record.keys) + 1, record.chain)
            record.keys.append(key)
            self._live[record.rank][key] += 1
            if record.active:
                self._active[record.rank][key] += 1

    def observe_outputs(
        self,
        outputs: Sequence[EngineCoreOutput],
        finished_requests: set[str] | None,
    ) -> None:
        for output in outputs:
            record = self._requests.get(output.request_id)
            if record is None:
                continue
            if not record.active:
                record.active = True
                self._increment(self._active[record.rank], record.keys)
            self._append_tokens(record, output.new_token_ids)

        for request_id in finished_requests or ():
            self.finish_request(request_id)

    def finish_request(self, request_id: str, *, keep_warm: bool = True) -> None:
        record = self._requests.pop(request_id, None)
        if record is None:
            return
        self._rank_work[record.rank] = max(
            0, self._rank_work[record.rank] - record.work_units
        )
        self._decrement(self._live[record.rank], record.keys)
        if record.active:
            self._decrement(self._active[record.rank], record.keys)

        if (
            keep_warm
            and self.warm_ttl_s > 0
            and self.max_warm_requests > 0
            and record.keys
        ):
            self._increment(self._warm[record.rank], record.keys)
            self._warm_requests.append(
                (self._clock() + self.warm_ttl_s, record.rank, record.keys)
            )
            while len(self._warm_requests) > self.max_warm_requests:
                self._drop_oldest_warm()

    @property
    def average_route_us(self) -> float:
        if not self.route_count:
            return 0.0
        return self.routing_time_ns / self.route_count / 1000

    @property
    def telemetry_snapshot(self) -> list[tuple[str, int, int, int, int, float]]:
        return [
            (
                state.graph_kind,
                state.graph_capacity,
                state.active_ctas,
                state.shared_ctas,
                state.singleton_ctas,
                round(state.kv_cache_usage, 4),
            )
            for state in self._engine_telemetry
        ]
