# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from vllm.v1.engine import EngineCoreEventType, FinishReason

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
    from vllm.v1.engine.kv_routing import GPUCacheRoutingIndex

PrefixKey = tuple[int, int]
RoutingPolicy = Literal["prefix_aware", "session_aware"]


@dataclass(slots=True)
class _Prefix:
    chain: int
    keys: list[PrefixKey]
    tail: list[int]
    num_blocks: int


@dataclass(slots=True)
class _RequestPrefix:
    rank: int
    prefix: _Prefix
    work_units: int
    session_id: str | None = None
    resident: bool = False
    uncomputed_token_ids: list[int] = field(default_factory=list)


class PrefixAwareDPRouter:
    """Agentrix routing policies for vLLM's internal DP balancer.

    Physical KV ownership remains entirely inside each engine. The router only
    remembers where eligible requests ran and keeps finished prefixes warm for
    a bounded period. Session-aware routing balances first turns and applies
    prefix affinity to follow-up turns. Load limits always take precedence.
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
        max_warm_checkpoints: int = 262144,
        max_checkpoints_per_request: int = 256,
        checkpoint_stride_blocks: int = 4,
        work_slack_tokens: int = 8192,
        decode_token_weight: int = 16,
        routing_policy: RoutingPolicy = "prefix_aware",
        session_overload_ratio: float = 2.0,
        session_hit_ratio: float = 0.5,
        use_kv_events: bool = False,
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
        if max_warm_checkpoints < 0:
            raise ValueError("max_warm_checkpoints must be non-negative")
        if max_checkpoints_per_request <= 0:
            raise ValueError("max_checkpoints_per_request must be positive")
        if checkpoint_stride_blocks <= 0:
            raise ValueError("checkpoint_stride_blocks must be positive")
        if work_slack_tokens < 0:
            raise ValueError("work_slack_tokens must be non-negative")
        if decode_token_weight <= 0:
            raise ValueError("decode_token_weight must be positive")
        if routing_policy not in ("prefix_aware", "session_aware"):
            raise ValueError(f"unsupported DP routing policy: {routing_policy!r}")
        if not math.isfinite(session_overload_ratio) or session_overload_ratio < 1:
            raise ValueError("session_overload_ratio must be finite and at least 1")
        if not math.isfinite(session_hit_ratio) or not 0 <= session_hit_ratio <= 1:
            raise ValueError("session_hit_ratio must be finite and in [0, 1]")

        self.num_ranks = num_ranks
        self.block_size = block_size
        self.load_slack = load_slack
        self.warm_ttl_s = warm_ttl_s
        self.min_prefix_blocks = min_prefix_blocks
        self.max_warm_requests = max_warm_requests
        self.max_warm_checkpoints = max_warm_checkpoints
        self.max_checkpoints_per_request = max_checkpoints_per_request
        self.checkpoint_stride_blocks = checkpoint_stride_blocks
        self.work_slack_tokens = work_slack_tokens
        self.decode_token_weight = decode_token_weight
        self.routing_policy = routing_policy
        self.session_overload_ratio = session_overload_ratio
        self.session_hit_ratio = session_hit_ratio
        self._clock = clock
        self._cache_index: GPUCacheRoutingIndex | None = None
        if use_kv_events:
            from vllm.v1.engine.kv_routing import GPUCacheRoutingIndex

            self._cache_index = GPUCacheRoutingIndex(
                num_ranks, block_size, max(1, max_warm_checkpoints // num_ranks)
            )

        self._requests: dict[str, _RequestPrefix] = {}
        self._pending_prefixes: dict[str, _Prefix] = {}
        self._pending_work: dict[str, int] = {}
        self._rank_work = [0] * num_ranks
        self._resident = [Counter[PrefixKey]() for _ in range(num_ranks)]
        self._warm_requests: deque[tuple[float, int, list[PrefixKey]]] = deque()
        self._num_warm_checkpoints = 0
        self._session_routes: OrderedDict[str, tuple[float, int]] = OrderedDict()

        self.route_count = 0
        self.affinity_route_count = 0
        self.rank_route_counts = [0] * num_ranks
        self.routing_time_ns = 0
        self.last_affinity_blocks = 0
        self.first_turn_balance_count = 0
        self.followup_affinity_count = 0
        self.followup_rebalance_count = 0
        self.session_overload_rebalance_count = 0
        self.session_cache_miss_rebalance_count = 0
        self.session_id_route_count = 0
        self.session_prefix_fallback_count = 0
        self.unknown_turn_route_count = 0
        self.cache_event_route_count = 0

    @staticmethod
    def _namespace(request: EngineCoreRequest) -> int | None:
        # Avoid false affinity when the frontend cannot cheaply reproduce the
        # physical cache identity used for prompt embeddings.
        if request.prompt_embeds is not None:
            return None
        if request.prompt_is_token_ids is not None and not all(
            request.prompt_is_token_ids
        ):
            return None

        mm_identity: list[tuple[str, str, int, int]] = []
        for feature in request.mm_features or ():
            if feature.identifier is None:
                return None
            mm_identity.append(
                (
                    feature.modality,
                    feature.identifier,
                    feature.mm_position.offset,
                    feature.mm_position.length,
                )
            )
        lora_name = (
            request.lora_request.lora_name if request.lora_request is not None else None
        )
        return hash((request.cache_salt, lora_name, tuple(mm_identity)))

    def should_route(self, request: EngineCoreRequest) -> bool:
        """Cheap eligibility check used to preserve the native DP fast path."""
        if getattr(request, "resumable", False):
            return False
        token_ids = request.prompt_token_ids
        if (
            token_ids is None
            or len(token_ids) < self.min_prefix_blocks * self.block_size
        ):
            return False
        sampling_params = request.sampling_params
        if sampling_params is not None:
            skip_cache = getattr(sampling_params, "skip_reading_prefix_cache", None)
            if skip_cache is not None:
                return not skip_cache and self._namespace(request) is not None
        pooling_params = request.pooling_params
        if pooling_params is not None and getattr(
            pooling_params, "skip_reading_prefix_cache", False
        ):
            return False
        return self._namespace(request) is not None

    def _limit_checkpoints(self, keys: list[PrefixKey]) -> list[PrefixKey]:
        if len(keys) <= self.max_checkpoints_per_request:
            return keys
        minimum = next((key for key in keys if key[0] == self.min_prefix_blocks), None)
        if minimum is None or self.max_checkpoints_per_request == 1:
            return keys[-self.max_checkpoints_per_request :]
        newest = keys[-(self.max_checkpoints_per_request - 1) :]
        if minimum in newest:
            return newest
        return [minimum, *newest]

    def _build_prefix(self, request: EngineCoreRequest) -> _Prefix | None:
        token_ids = request.prompt_token_ids
        namespace = self._namespace(request)
        if token_ids is None or namespace is None:
            return None

        num_blocks = len(token_ids) // self.block_size
        chain = namespace
        keys: list[PrefixKey] = []
        for block_index in range(num_blocks):
            start = block_index * self.block_size
            block = tuple(token_ids[start : start + self.block_size])
            chain = hash((chain, block))
            depth = block_index + 1
            if (
                depth % self.checkpoint_stride_blocks == 0
                or depth == self.min_prefix_blocks
                or depth == num_blocks
            ):
                keys.append((depth, chain))
        tail_start = num_blocks * self.block_size
        return _Prefix(
            chain,
            self._limit_checkpoints(keys),
            list(token_ids[tail_start:]),
            num_blocks,
        )

    @staticmethod
    def _increment(counter: Counter[PrefixKey], keys: Iterable[PrefixKey]) -> None:
        counter.update(keys)

    @staticmethod
    def _decrement(counter: Counter[PrefixKey], keys: Iterable[PrefixKey]) -> None:
        for key in keys:
            count = counter[key]
            if count <= 1:
                counter.pop(key, None)
            else:
                counter[key] = count - 1

    @staticmethod
    def _deepest_match(
        keys: Sequence[PrefixKey], counts: Counter[PrefixKey]
    ) -> tuple[int, int]:
        for key in reversed(keys):
            fanout = counts.get(key, 0)
            if fanout:
                return key[0], fanout
        return 0, 0

    def _drop_oldest_warm(self) -> None:
        _, rank, keys = self._warm_requests.popleft()
        self._decrement(self._resident[rank], keys)
        self._num_warm_checkpoints -= len(keys)

    def _expire_warm(self) -> None:
        now = self._clock()
        while self._warm_requests and self._warm_requests[0][0] <= now:
            self._drop_oldest_warm()
        while self._session_routes:
            _, (expires_at, _) = next(iter(self._session_routes.items()))
            if expires_at > now:
                break
            self._session_routes.popitem(last=False)

    def _remember_session(self, session_id: str, rank: int) -> None:
        if self.max_warm_requests == 0 or self.warm_ttl_s == 0:
            return
        self._session_routes.pop(session_id, None)
        self._session_routes[session_id] = (
            self._clock() + self.warm_ttl_s,
            rank,
        )
        while len(self._session_routes) > self.max_warm_requests:
            self._session_routes.popitem(last=False)

    def _candidate_work(
        self,
        request: EngineCoreRequest,
        cached_blocks: int,
        rank: int,
    ) -> int:
        prompt_tokens = len(request.prompt_token_ids or ())
        private_prompt_tokens = max(0, prompt_tokens - cached_blocks * self.block_size)
        max_tokens = int(getattr(request.sampling_params, "max_tokens", 1) or 1)
        return (
            self._rank_work[rank]
            + private_prompt_tokens
            + max_tokens * self.decode_token_weight
        )

    @staticmethod
    def _request_xarg(request: EngineCoreRequest, name: str) -> object | None:
        sampling_params = request.sampling_params
        extra_args = getattr(sampling_params, "extra_args", None)
        if not isinstance(extra_args, dict):
            return None
        return extra_args.get(name)

    @classmethod
    def _request_nonnegative_int(
        cls, request: EngineCoreRequest, name: str
    ) -> int | None:
        value = cls._request_xarg(request, name)
        if isinstance(value, bool):
            return None
        if not isinstance(value, (int, str, bytes, bytearray)):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    @classmethod
    def _request_session_id(cls, request: EngineCoreRequest) -> str | None:
        value = cls._request_xarg(request, "agentrix_session_id")
        if value is None or isinstance(value, bool):
            return None
        session_id = str(value).strip()
        return session_id or None

    def _session_affinity_allowed(
        self,
        request: EngineCoreRequest,
        rank: int,
        affinity_blocks: int,
        load_scores: Sequence[int],
        candidate_work: Sequence[int],
    ) -> tuple[bool, Literal["hit", "miss", "overload"]]:
        if affinity_blocks < self.min_prefix_blocks:
            return False, "miss"

        expected_history_tokens = self._request_nonnegative_int(
            request, "agentrix_history_tokens"
        )
        if expected_history_tokens is not None and (
            affinity_blocks * self.block_size
            < expected_history_tokens * self.session_hit_ratio
        ):
            return False, "miss"

        load_eligible = self._session_load_eligible_ranks(load_scores)
        load_overloaded = rank not in load_eligible
        work_overloaded = (
            candidate_work[rank]
            > min(candidate_work[candidate] for candidate in load_eligible)
            + self.work_slack_tokens
        )
        if load_overloaded or work_overloaded:
            return False, "overload"
        return True, "hit"

    def _session_load_eligible_ranks(self, load_scores: Sequence[int]) -> list[int]:
        mean_load = sum(load_scores) / self.num_ranks
        load_limit = self.session_overload_ratio * max(mean_load, 1.0)
        min_load = min(load_scores)
        return [
            rank
            for rank, load in enumerate(load_scores)
            if load <= load_limit and load <= min_load + self.load_slack
        ]

    def _finish_route(
        self,
        request: EngineCoreRequest,
        chosen_rank: int,
        chosen_depth: int,
        affinity_blocks: int,
        started_ns: int,
    ) -> int:
        self._pending_work[request.request_id] = (
            self._candidate_work(request, chosen_depth, chosen_rank)
            - self._rank_work[chosen_rank]
        )
        self.route_count += 1
        self.rank_route_counts[chosen_rank] += 1
        self.last_affinity_blocks = affinity_blocks
        self.routing_time_ns += time.perf_counter_ns() - started_ns
        return chosen_rank

    def choose_rank(
        self,
        request: EngineCoreRequest,
        engine_counts: Sequence[Sequence[int]],
        baseline_rank: int,
        start_index: int = 0,
    ) -> int:
        """Choose a prefix-affine rank within bounded load and work slack."""
        if len(engine_counts) != self.num_ranks:
            raise ValueError("engine count does not match the number of DP ranks")
        if not 0 <= baseline_rank < self.num_ranks:
            raise ValueError("baseline rank is out of range")

        started_ns = time.perf_counter_ns()
        self._expire_warm()
        prefix = self._pending_prefixes.get(request.request_id)
        if prefix is None:
            prefix = self._build_prefix(request)
        if prefix is None:
            return baseline_rank
        self._pending_prefixes[request.request_id] = prefix

        load_scores = [waiting * 4 + running for waiting, running in engine_counts]
        rank_matches = [
            self._deepest_match(prefix.keys, self._resident[rank])
            for rank in range(self.num_ranks)
        ]
        physical_matches: Sequence[int | None] = ()
        if self._cache_index is not None and self._cache_index.supports(request):
            physical_matches = self._cache_index.lookup(request.prompt_token_ids or ())
            for rank, cached_blocks in enumerate(physical_matches):
                if cached_blocks is not None:
                    rank_matches[rank] = (cached_blocks, rank_matches[rank][1])
            self.cache_event_route_count += int(
                any(match is not None for match in physical_matches)
            )
        candidate_work = [
            self._candidate_work(request, rank_matches[rank][0], rank)
            for rank in range(self.num_ranks)
        ]

        session_turn = self._request_nonnegative_int(request, "agentrix_turn")
        beam_step = self._request_nonnegative_int(request, "agentrix_beam_step")
        if (
            self.routing_policy == "session_aware"
            and session_turn == 0
            and (beam_step is None or beam_step == 0)
        ):
            self.first_turn_balance_count += 1
            return self._finish_route(
                request,
                baseline_rank,
                rank_matches[baseline_rank][0],
                rank_matches[baseline_rank][0],
                started_ns,
            )

        if self.routing_policy == "session_aware" and session_turn is not None:
            session_id = self._request_session_id(request)
            session_route = (
                self._session_routes.get(session_id)
                if session_id is not None and (beam_step is None or beam_step == 0)
                else None
            )
            if session_route is not None:
                assert session_id is not None
                best_rank = session_route[1]
                # A live session mapping may outlast GPU residency. Prefer a
                # verified longer GPU prefix before applying the load guards.
                for rank, cached_blocks in enumerate(physical_matches):
                    if (
                        cached_blocks is not None
                        and cached_blocks > rank_matches[best_rank][0]
                    ):
                        best_rank = rank
                self.session_id_route_count += 1
                self._remember_session(session_id, best_rank)
            else:
                best_rank = max(
                    range(self.num_ranks),
                    key=lambda rank: (
                        rank_matches[rank],
                        -candidate_work[rank],
                        -load_scores[rank],
                        -((rank - start_index) % self.num_ranks),
                    ),
                )
                self.session_prefix_fallback_count += 1
            affinity_blocks = rank_matches[best_rank][0]
            allowed, reason = self._session_affinity_allowed(
                request,
                best_rank,
                affinity_blocks,
                load_scores,
                candidate_work,
            )
            if allowed:
                chosen_rank = best_rank
            elif reason == "overload":
                load_eligible = self._session_load_eligible_ranks(load_scores)
                chosen_rank = min(
                    load_eligible,
                    key=lambda rank: (
                        candidate_work[rank],
                        load_scores[rank],
                        (rank - start_index) % self.num_ranks,
                    ),
                )
            else:
                chosen_rank = baseline_rank
            if allowed:
                self.affinity_route_count += 1
                self.followup_affinity_count += 1
            else:
                self.followup_rebalance_count += 1
                if reason == "overload":
                    self.session_overload_rebalance_count += 1
                else:
                    self.session_cache_miss_rebalance_count += 1
            return self._finish_route(
                request,
                chosen_rank,
                rank_matches[chosen_rank][0],
                affinity_blocks,
                started_ns,
            )

        if self.routing_policy == "session_aware":
            self.unknown_turn_route_count += 1

        min_load = min(load_scores)
        load_eligible = [
            rank
            for rank, load in enumerate(load_scores)
            if load <= min_load + self.load_slack
        ]

        min_work = min(candidate_work[rank] for rank in load_eligible)
        work_eligible = [
            rank
            for rank in load_eligible
            if candidate_work[rank] <= min_work + self.work_slack_tokens
        ]
        best_rank = max(
            work_eligible,
            key=lambda rank: (
                rank_matches[rank],
                -candidate_work[rank],
                -load_scores[rank],
                -((rank - start_index) % self.num_ranks),
            ),
        )
        affinity_blocks = rank_matches[best_rank][0]
        chosen_rank = baseline_rank
        if affinity_blocks >= self.min_prefix_blocks:
            chosen_rank = best_rank
            self.affinity_route_count += 1

        return self._finish_route(
            request,
            chosen_rank,
            rank_matches[chosen_rank][0],
            affinity_blocks,
            started_ns,
        )

    def add_request(self, request: EngineCoreRequest, rank: int) -> None:
        prefix = self._pending_prefixes.pop(request.request_id, None)
        if prefix is None:
            prefix = self._build_prefix(request)
        if prefix is None:
            return
        if request.request_id in self._requests:
            self.finish_request(request.request_id, keep_warm=False)
        work_units = self._pending_work.pop(
            request.request_id,
            self._candidate_work(request, 0, rank) - self._rank_work[rank],
        )
        beam_step = self._request_nonnegative_int(request, "agentrix_beam_step")
        session_id = self._request_session_id(request)
        if beam_step is not None and beam_step > 0:
            if session_id is not None:
                self._session_routes.pop(session_id, None)
            session_id = None
        self._requests[request.request_id] = _RequestPrefix(
            rank, prefix, work_units, session_id=session_id
        )
        self._rank_work[rank] += work_units
        if self.routing_policy == "session_aware" and session_id is not None:
            self._remember_session(session_id, rank)

    def _append_computed_tokens(
        self, record: _RequestPrefix, token_ids: Sequence[int]
    ) -> None:
        if not token_ids:
            return
        prefix = record.prefix
        previous_keys = prefix.keys
        keys = [
            key
            for key in previous_keys
            if key[0] != prefix.num_blocks
            or key[0] == self.min_prefix_blocks
            or key[0] % self.checkpoint_stride_blocks == 0
        ]
        prefix.tail.extend(token_ids)
        while len(prefix.tail) >= self.block_size:
            block = tuple(prefix.tail[: self.block_size])
            del prefix.tail[: self.block_size]
            prefix.chain = hash((prefix.chain, block))
            prefix.num_blocks += 1
            key = (prefix.num_blocks, prefix.chain)
            if (
                prefix.num_blocks == self.min_prefix_blocks
                or prefix.num_blocks % self.checkpoint_stride_blocks == 0
            ):
                keys.append(key)
        if prefix.num_blocks and (not keys or keys[-1][0] != prefix.num_blocks):
            keys.append((prefix.num_blocks, prefix.chain))
        prefix.keys = self._limit_checkpoints(keys)
        if record.resident:
            previous = set(previous_keys)
            current = set(prefix.keys)
            self._decrement(self._resident[record.rank], previous - current)
            self._increment(self._resident[record.rank], current - previous)

    def _observe_generated_tokens(
        self, record: _RequestPrefix, token_ids: Sequence[int]
    ) -> None:
        if not token_ids:
            return
        computed = [*record.uncomputed_token_ids, *token_ids[:-1]]
        record.uncomputed_token_ids = [token_ids[-1]]
        self._append_computed_tokens(record, computed)

    def invalidate_requests(self, request_ids: Iterable[str]) -> None:
        """Discard active residency hints for requests whose blocks were freed."""
        for request_id in request_ids:
            record = self._requests.get(request_id)
            if record is not None and record.resident:
                self._decrement(self._resident[record.rank], record.prefix.keys)
                record.resident = False

    def observe_outputs(
        self,
        outputs: Sequence[EngineCoreOutput],
        finished_requests: set[str] | None,
        preempted_requests: set[str] | None = None,
    ) -> None:
        # A preemption means the scheduler freed all blocks for the request.
        # Treat this out-of-band signal as authoritative for the whole output
        # batch, even when an in-flight model execution also sampled tokens.
        forced_preempted = set(preempted_requests or ())
        nonresident_requests = set(forced_preempted)
        self.invalidate_requests(forced_preempted)
        for output in outputs:
            record = self._requests.get(output.request_id)
            if record is None:
                continue
            events = output.events or ()
            # Events describe request history, so the latest scheduling
            # transition determines whether a prior preemption is still in
            # effect. A SCHEDULED event alone is not proof that model work has
            # completed; require an actual result before restoring residency.
            last_preempted = max(
                (
                    index
                    for index, event in enumerate(events)
                    if event.type == EngineCoreEventType.PREEMPTED
                ),
                default=-1,
            )
            last_scheduled = max(
                (
                    index
                    for index, event in enumerate(events)
                    if event.type == EngineCoreEventType.SCHEDULED
                ),
                default=-1,
            )
            event_preempted = last_preempted > last_scheduled
            event_resumed = last_preempted >= 0 and last_scheduled > last_preempted
            has_model_output = bool(output.new_token_ids) or any(
                getattr(output, name, None) is not None
                for name in ("pooling_output", "new_prompt_logprobs_tensors")
            )
            must_remain_nonresident = (
                output.request_id in forced_preempted
                or event_preempted
                or (event_resumed and not has_model_output)
            )
            if must_remain_nonresident:
                nonresident_requests.add(output.request_id)
                self.invalidate_requests((output.request_id,))
                # The sampled tokens are still part of the logical session,
                # but none of its prefix may be advertised as resident until
                # a later post-resume output confirms recomputation.
                self._observe_generated_tokens(record, output.new_token_ids)
                continue
            if event_resumed:
                nonresident_requests.discard(output.request_id)
            # Direct abort/error outputs may arrive before the request ever
            # executes. Do not turn those bookkeeping-only outputs into a
            # physical-residency hint.
            if (
                not record.resident
                and getattr(output, "finish_reason", None)
                in (FinishReason.ABORT, FinishReason.ERROR)
                and not output.new_token_ids
            ):
                continue
            if not record.resident:
                record.resident = True
                self._increment(self._resident[record.rank], record.prefix.keys)
            self._observe_generated_tokens(record, output.new_token_ids)

        for request_id in finished_requests or ():
            self.finish_request(
                request_id, keep_warm=request_id not in nonresident_requests
            )

    def finish_request(self, request_id: str, *, keep_warm: bool = True) -> None:
        record = self._requests.pop(request_id, None)
        if record is None:
            return
        rank = record.rank
        keys = record.prefix.keys
        self._rank_work[rank] = max(0, self._rank_work[rank] - record.work_units)

        can_keep_warm = (
            keep_warm
            and record.resident
            and self.warm_ttl_s > 0
            and self.max_warm_requests > 0
            and self.max_warm_checkpoints > 0
            and bool(keys)
        )
        if can_keep_warm:
            self._warm_requests.append((self._clock() + self.warm_ttl_s, rank, keys))
            self._num_warm_checkpoints += len(keys)
            if self.routing_policy == "session_aware" and record.session_id is not None:
                self._remember_session(record.session_id, rank)
            while (
                len(self._warm_requests) > self.max_warm_requests
                or self._num_warm_checkpoints > self.max_warm_checkpoints
            ):
                self._drop_oldest_warm()
        elif record.resident:
            self._decrement(self._resident[rank], keys)

    def invalidate_residency(self) -> None:
        """Discard physical-cache hints after a successful cache reset."""
        self._resident = [Counter() for _ in range(self.num_ranks)]
        self._warm_requests.clear()
        self._num_warm_checkpoints = 0
        self._session_routes.clear()
        if self._cache_index is not None:
            self._cache_index.clear()
        for record in self._requests.values():
            record.resident = False

    def observe_cache_events(self, rank: int, payload: bytes | None) -> None:
        """Update GPU location hints from one engine's ordered event batch."""
        if self._cache_index is not None and payload is not None:
            self._cache_index.update(rank, payload)

    @property
    def average_route_us(self) -> float:
        if not self.route_count:
            return 0.0
        return self.routing_time_ns / self.route_count / 1000
