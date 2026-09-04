# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-overhead, physical-block KV residency tracking.

The index in this module is deliberately policy-free. It mirrors block-pool
state so eviction and offload policies can be evaluated in shadow mode before
they are allowed to affect allocation decisions.
"""

import time
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import TYPE_CHECKING

from vllm import envs

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool


class KVBlockState(IntEnum):
    """Lifecycle segment for a physical KV cache block."""

    UNTRACKED = 0
    FREE = 1
    UNHASHED = 2
    ACTIVE = 3
    ACTIVE_SHARED = 4
    WARM = 5
    WARM_SHARED = 6
    COOLING = 7
    COOLING_SHARED = 8
    COLD = 9
    COLD_SHARED = 10


class KVResidencyTier(IntFlag):
    """Known copies of the generation currently assigned to a block ID."""

    NONE = 0
    GPU = 1
    CPU = 2
    REMOTE = 4


@dataclass(frozen=True, slots=True)
class KVResidencyEntry:
    """Debug snapshot for one physical block."""

    block_id: int
    generation: int
    state: KVBlockState
    ref_count: int
    request_ref_count: int
    reuse_count: int
    peak_fanout: int
    hash_count: int
    residency: KVResidencyTier
    backing_up: bool
    backing_up_tiers: KVResidencyTier
    last_access_ns: int


@dataclass(frozen=True, slots=True)
class KVResidencyStats:
    """Constant-size shadow-mode residency snapshot."""

    epoch: int = 0
    tracked_blocks: int = 0
    free_blocks: int = 0
    unhashed_blocks: int = 0
    active_blocks: int = 0
    active_shared_blocks: int = 0
    warm_blocks: int = 0
    warm_shared_blocks: int = 0
    cooling_blocks: int = 0
    cooling_shared_blocks: int = 0
    cold_blocks: int = 0
    cold_shared_blocks: int = 0
    shared_blocks: int = 0
    backing_up_blocks: int = 0
    backed_up_blocks: int = 0
    allocations: int = 0
    cache_hits: int = 0
    cache_insertions: int = 0
    cache_removals: int = 0
    cache_evictions: int = 0
    shared_cache_evictions: int = 0
    unbacked_cache_evictions: int = 0
    inflight_cache_evictions: int = 0
    aging_transitions: int = 0
    stale_updates: int = 0


@dataclass(slots=True)
class _BlockResidency:
    """Mutable hot-path metadata; slots avoid per-block dictionaries."""

    state: int = KVBlockState.FREE
    generation: int = 0
    ref_count: int = 0
    request_ref_count: int = 0
    reuse_count: int = 0
    peak_fanout: int = 0
    hash_count: int = 0
    residency: int = KVResidencyTier.NONE
    last_access_ns: int = 0
    prev: int = -1
    next: int = -1


_SHARED_STATES = (
    KVBlockState.ACTIVE_SHARED,
    KVBlockState.WARM_SHARED,
    KVBlockState.COOLING_SHARED,
    KVBlockState.COLD_SHARED,
)
_IDLE_STATES = (
    KVBlockState.WARM,
    KVBlockState.WARM_SHARED,
    KVBlockState.COOLING,
    KVBlockState.COOLING_SHARED,
    KVBlockState.COLD,
    KVBlockState.COLD_SHARED,
)
_IS_IDLE_STATE = tuple(state in _IDLE_STATES for state in KVBlockState)
_TIER_NONE = int(KVResidencyTier.NONE)
_TIER_GPU = int(KVResidencyTier.GPU)
_TIER_CPU = int(KVResidencyTier.CPU)
_TIER_REMOTE = int(KVResidencyTier.REMOTE)
_LOWER_TIERS = _TIER_CPU | _TIER_REMOTE
_AGING_SEGMENTS = (
    (KVBlockState.WARM, KVBlockState.COOLING),
    (KVBlockState.WARM_SHARED, KVBlockState.COOLING_SHARED),
    (KVBlockState.COOLING, KVBlockState.COLD),
    (KVBlockState.COOLING_SHARED, KVBlockState.COLD_SHARED),
)


class KVResidencyIndex:
    """Shadow index keyed by physical block ID.

    Intrusive links are used only by idle cache segments. Active, unhashed,
    and free blocks never participate in victim ordering. Moving an idle block
    is O(1), and aging only consumes ordered segment heads, so the scheduler
    path never performs a cache-wide scan.

    Time is sampled once per scheduler step by :meth:`advance`. Per-block
    callbacks reuse that cached timestamp instead of invoking a clock.
    """

    __slots__ = (
        "num_blocks",
        "null_block_id",
        "shared_reuse_threshold",
        "aging_budget",
        "aging_cursor",
        "warm_ns",
        "cooling_ns",
        "epoch",
        "now_ns",
        "entries",
        "heads",
        "tails",
        "counts",
        "allocations",
        "cache_hits",
        "cache_insertions",
        "cache_removals",
        "cache_evictions",
        "shared_cache_evictions",
        "unbacked_cache_evictions",
        "inflight_cache_evictions",
        "aging_transitions",
        "stale_updates",
        "_backing_up_count",
        "_backed_up_count",
        "_inflight",
        "_restore_positions",
    )

    def __init__(
        self,
        num_blocks: int,
        *,
        null_block_id: int,
        warm_seconds: float = 10.0,
        cooling_seconds: float = 100.0,
        shared_reuse_threshold: int = 2,
        aging_budget: int = 256,
        now_ns: int | None = None,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if not 0 <= warm_seconds <= cooling_seconds:
            raise ValueError(
                "expected 0 <= warm_seconds <= cooling_seconds, got "
                f"{warm_seconds} and {cooling_seconds}"
            )
        if shared_reuse_threshold <= 0:
            raise ValueError("shared_reuse_threshold must be positive")
        if aging_budget <= 0:
            raise ValueError("aging_budget must be positive")
        if not 0 <= null_block_id < num_blocks:
            raise ValueError("null_block_id must identify a physical block")

        self.num_blocks = num_blocks
        self.null_block_id = null_block_id
        self.shared_reuse_threshold = shared_reuse_threshold
        self.aging_budget = aging_budget
        self.aging_cursor = 0
        self.warm_ns = int(warm_seconds * 1e9)
        self.cooling_ns = int(cooling_seconds * 1e9)
        self.epoch = 0
        self.now_ns = time.monotonic_ns() if now_ns is None else now_ns

        # A slotted object is slightly larger than packed arrays, but avoids
        # Python's array boxing cost on every hot-path field access. No hash map
        # is duplicated: logical-key lookup remains owned by BlockPool.
        self.entries = [
            _BlockResidency(last_access_ns=self.now_ns) for _ in range(num_blocks)
        ]
        self.entries[null_block_id].state = KVBlockState.UNTRACKED
        num_states = len(KVBlockState)
        self.heads = [-1] * num_states
        self.tails = [-1] * num_states
        self.counts = [0] * num_states
        self.counts[KVBlockState.UNTRACKED] = 1
        self.counts[KVBlockState.FREE] = num_blocks - 1

        self.allocations = 0
        self.cache_hits = 0
        self.cache_insertions = 0
        self.cache_removals = 0
        self.cache_evictions = 0
        self.shared_cache_evictions = 0
        self.unbacked_cache_evictions = 0
        self.inflight_cache_evictions = 0
        self.aging_transitions = 0
        self.stale_updates = 0
        self._backing_up_count = 0
        self._backed_up_count = 0
        self._inflight: dict[int, set[tuple[int, int]]] = {}
        self._restore_positions: dict[int, tuple[int, int, int, int, int]] = {}

    def _append_idle(
        self,
        block_id: int,
        entry: _BlockResidency,
        state_idx: int,
    ) -> None:
        tail = self.tails[state_idx]
        entry.prev = tail
        entry.next = -1
        if tail < 0:
            self.heads[state_idx] = block_id
        else:
            self.entries[tail].next = block_id
        self.tails[state_idx] = block_id

    def _unlink_idle(self, entry: _BlockResidency) -> None:
        state_idx = entry.state
        if entry.prev < 0:
            self.heads[state_idx] = entry.next
        else:
            self.entries[entry.prev].next = entry.next
        if entry.next < 0:
            self.tails[state_idx] = entry.prev
        else:
            self.entries[entry.next].prev = entry.prev
        entry.prev = -1
        entry.next = -1

    def _clear_restore_position(self, block_id: int) -> None:
        self._restore_positions.pop(block_id, None)

    def _save_restore_position(
        self,
        block_id: int,
        entry: _BlockResidency,
    ) -> None:
        prev_id = entry.prev
        next_id = entry.next
        prev_generation = self.entries[prev_id].generation if prev_id >= 0 else 0
        next_generation = self.entries[next_id].generation if next_id >= 0 else 0
        self._restore_positions[block_id] = (
            entry.state,
            prev_id,
            prev_generation,
            next_id,
            next_generation,
        )

    def _restore_idle_position(
        self,
        block_id: int,
        entry: _BlockResidency,
    ) -> bool:
        """Restore a temporarily pinned entry without searching its queue."""
        position = self._restore_positions.get(block_id)
        if position is None:
            return False
        state_idx, prev_id, prev_generation, next_id, next_generation = position

        if prev_id < 0:
            anchors_match = self.heads[state_idx] == next_id
        else:
            prev_entry = self.entries[prev_id]
            anchors_match = (
                prev_entry.generation == prev_generation
                and prev_entry.state == state_idx
                and prev_entry.next == next_id
            )
        if next_id >= 0:
            next_entry = self.entries[next_id]
            anchors_match &= (
                next_entry.generation == next_generation
                and next_entry.state == state_idx
                and next_entry.prev == prev_id
            )
        else:
            anchors_match &= self.tails[state_idx] == prev_id
        if not anchors_match:
            return False

        old_state_idx = entry.state
        self.counts[old_state_idx] -= 1
        entry.state = state_idx
        self.counts[state_idx] += 1
        entry.prev = prev_id
        entry.next = next_id
        if prev_id < 0:
            self.heads[state_idx] = block_id
        else:
            self.entries[prev_id].next = block_id
        if next_id < 0:
            self.tails[state_idx] = block_id
        else:
            self.entries[next_id].prev = block_id
        self._clear_restore_position(block_id)
        return True

    def _move(
        self,
        block_id: int,
        entry: _BlockResidency,
        new_state: KVBlockState,
    ) -> None:
        old_state_idx = entry.state
        new_state_idx = int(new_state)
        if old_state_idx == new_state_idx:
            return
        if _IS_IDLE_STATE[old_state_idx]:
            self._unlink_idle(entry)
        self.counts[old_state_idx] -= 1
        entry.state = new_state_idx
        self.counts[new_state_idx] += 1
        if _IS_IDLE_STATE[new_state_idx]:
            self._append_idle(block_id, entry, new_state_idx)

    @staticmethod
    def _next_generation(entry: _BlockResidency) -> int:
        generation = (entry.generation + 1) & 0xFFFFFFFF
        # Keep zero as the sentinel for a block that has never carried data.
        if generation == 0:
            generation = 1
        entry.generation = generation
        return generation

    def _is_shared(self, entry: _BlockResidency) -> bool:
        return (
            entry.peak_fanout >= 2 or entry.reuse_count >= self.shared_reuse_threshold
        )

    def _active_state(self, entry: _BlockResidency) -> KVBlockState:
        if entry.hash_count == 0:
            return KVBlockState.UNHASHED
        if self._is_shared(entry):
            return KVBlockState.ACTIVE_SHARED
        return KVBlockState.ACTIVE

    def _idle_state(self, entry: _BlockResidency) -> KVBlockState:
        if self._is_shared(entry):
            return KVBlockState.WARM_SHARED
        return KVBlockState.WARM

    def _aged_idle_state(self, entry: _BlockResidency) -> KVBlockState:
        age_ns = self.now_ns - entry.last_access_ns
        shared = self._is_shared(entry)
        if age_ns >= self.cooling_ns:
            return KVBlockState.COLD_SHARED if shared else KVBlockState.COLD
        if age_ns >= self.warm_ns:
            return KVBlockState.COOLING_SHARED if shared else KVBlockState.COOLING
        return KVBlockState.WARM_SHARED if shared else KVBlockState.WARM

    @staticmethod
    def _reset_generation_metadata(entry: _BlockResidency) -> None:
        entry.reuse_count = 0
        entry.peak_fanout = entry.request_ref_count

    def on_allocated(self, block_id: int, ref_count: int) -> int:
        """Start a new physical-block generation after allocation."""
        entry = self.entries[block_id]
        if entry.state == KVBlockState.UNTRACKED:
            return 0
        generation = self._next_generation(entry)
        self._clear_restore_position(block_id)
        self._clear_transfer_state(block_id, entry)
        entry.ref_count = ref_count
        entry.request_ref_count = ref_count
        self._reset_generation_metadata(entry)
        entry.hash_count = 0
        entry.residency = _TIER_GPU
        entry.last_access_ns = self.now_ns
        self._move(block_id, entry, KVBlockState.UNHASHED)
        self.allocations += 1
        return generation

    def on_cache_inserted(self, block_id: int) -> None:
        """Record a new prefix-cache key for the current generation."""
        entry = self.entries[block_id]
        if entry.state == KVBlockState.UNTRACKED:
            return
        entry.hash_count += 1
        entry.residency |= _TIER_GPU
        state = (
            self._active_state(entry)
            if entry.ref_count > 0
            else self._idle_state(entry)
        )
        self._move(block_id, entry, state)
        self.cache_insertions += 1

    def on_cache_hit(self, block_id: int, ref_count: int) -> None:
        """Record a request adopting a cached block."""
        entry = self.entries[block_id]
        if entry.state == KVBlockState.UNTRACKED:
            return
        entry.ref_count = ref_count
        entry.request_ref_count += 1
        entry.reuse_count += 1
        self._clear_restore_position(block_id)
        if entry.request_ref_count > entry.peak_fanout:
            entry.peak_fanout = entry.request_ref_count
        entry.last_access_ns = self.now_ns
        self._move(block_id, entry, self._active_state(entry))
        self.cache_hits += 1

    def on_pinned(self, block_id: int, ref_count: int) -> None:
        """Mirror a temporary internal reference without recording reuse."""
        entry = self.entries[block_id]
        if entry.state == KVBlockState.UNTRACKED:
            return
        if _IS_IDLE_STATE[entry.state]:
            self._save_restore_position(block_id, entry)
        entry.ref_count = ref_count
        self._move(block_id, entry, self._active_state(entry))

    def on_released(self, block_id: int, ref_count: int) -> None:
        """Mirror a reference-count decrement from BlockPool.free_blocks."""
        entry = self.entries[block_id]
        if entry.state == KVBlockState.UNTRACKED:
            return
        entry.ref_count = ref_count
        if entry.request_ref_count > 0:
            entry.request_ref_count -= 1
        else:
            self.stale_updates += 1
        if entry.request_ref_count == 0:
            entry.last_access_ns = self.now_ns
        if ref_count > 0:
            self._move(block_id, entry, self._active_state(entry))
            return
        self._clear_restore_position(block_id)
        if entry.hash_count > 0:
            self._move(block_id, entry, self._idle_state(entry))
        else:
            entry.residency = _TIER_NONE
            self._move(block_id, entry, KVBlockState.FREE)

    def on_unpinned(self, block_id: int, ref_count: int) -> None:
        """Mirror release of a temporary reference without refreshing age."""
        entry = self.entries[block_id]
        if entry.state == KVBlockState.UNTRACKED:
            return
        entry.ref_count = ref_count
        if ref_count > 0:
            self._move(block_id, entry, self._active_state(entry))
        elif entry.hash_count > 0:
            if self._restore_idle_position(block_id, entry):
                return
            # The saved neighbors can move while a long-running pin is held.
            # Treat that uncommon case as a fresh idle insertion so the queue
            # remains time ordered without an O(N) search on the scheduler path.
            entry.last_access_ns = self.now_ns
            self._clear_restore_position(block_id)
            self._move(block_id, entry, self._idle_state(entry))
        else:
            self._clear_restore_position(block_id)
            entry.residency = _TIER_NONE
            self._move(block_id, entry, KVBlockState.FREE)

    def on_cache_removed(
        self,
        block_id: int,
        num_hashes: int,
        *,
        evicted: bool = False,
    ) -> None:
        """Invalidate local keys and any in-flight backup for those keys."""
        entry = self.entries[block_id]
        if entry.state == KVBlockState.UNTRACKED or num_hashes <= 0:
            return
        if evicted:
            self.cache_evictions += 1
            self.shared_cache_evictions += self._is_shared(entry)
            self.unbacked_cache_evictions += not bool(entry.residency & _LOWER_TIERS)
            self.inflight_cache_evictions += block_id in self._inflight
        self._next_generation(entry)
        self._clear_restore_position(block_id)
        self._reset_generation_metadata(entry)
        entry.hash_count = 0
        self._clear_transfer_state(block_id, entry)
        entry.residency = _TIER_GPU if entry.ref_count > 0 else _TIER_NONE
        target = KVBlockState.UNHASHED if entry.ref_count > 0 else KVBlockState.FREE
        self._move(block_id, entry, target)
        self.cache_removals += num_hashes

    def advance(self, now_ns: int | None = None) -> int:
        """Advance time and perform at most ``aging_budget`` transitions."""
        new_now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if new_now_ns < self.now_ns:
            raise ValueError("residency clock cannot move backwards")
        self.now_ns = new_now_ns
        self.epoch += 1

        cutoffs = (
            new_now_ns - self.warm_ns,
            new_now_ns - self.warm_ns,
            new_now_ns - self.cooling_ns,
            new_now_ns - self.cooling_ns,
        )
        remaining = self.aging_budget
        transitioned = 0
        for offset in range(len(_AGING_SEGMENTS)):
            segment_idx = (self.aging_cursor + offset) % len(_AGING_SEGMENTS)
            source, target = _AGING_SEGMENTS[segment_idx]
            moved = self._age_segment(
                source,
                target,
                cutoffs[segment_idx],
                remaining,
            )
            transitioned += moved
            remaining -= moved
            if remaining == 0:
                self.aging_cursor = (segment_idx + 1) % len(_AGING_SEGMENTS)
                break
        else:
            self.aging_cursor = (self.aging_cursor + 1) % len(_AGING_SEGMENTS)
        return transitioned

    def on_step(self) -> None:
        """Implement the observer scheduler-step notification."""
        self.advance()

    def _age_segment(
        self,
        source: KVBlockState,
        target: KVBlockState,
        cutoff_ns: int,
        limit: int,
    ) -> int:
        block_id = self.heads[source]
        transitioned = 0
        while block_id >= 0 and transitioned < limit:
            entry = self.entries[block_id]
            if entry.last_access_ns > cutoff_ns:
                break
            self._move(block_id, entry, target)
            self.aging_transitions += 1
            transitioned += 1
            block_id = self.heads[source]
        return transitioned

    def start_backup(
        self,
        block_id: int,
        generation: int,
        tier: KVResidencyTier,
        operation_id: int,
    ) -> bool:
        """Register one tier-specific asynchronous backup operation."""
        tier_value = self._validate_backup_tier(tier)
        entry = self.entries[block_id]
        if not self._matches_cached_generation(entry, generation):
            self.stale_updates += 1
            return False
        operation = (tier_value, operation_id)
        operations = self._inflight.get(block_id)
        if operations is not None and operation in operations:
            self.stale_updates += 1
            return False
        if operations is None:
            operations = self._inflight[block_id] = set()
            self._backing_up_count += 1
        operations.add(operation)
        return True

    def finish_backup(
        self,
        block_id: int,
        generation: int,
        tier: KVResidencyTier,
        operation_id: int,
        *,
        success: bool,
    ) -> bool:
        """Apply an asynchronous backup ACK if its generation is still live."""
        tier_value = self._validate_backup_tier(tier)
        entry = self.entries[block_id]
        if not self._matches_cached_generation(entry, generation):
            self.stale_updates += 1
            return False
        operations = self._inflight.get(block_id)
        operation = (tier_value, operation_id)
        if operations is None or operation not in operations:
            self.stale_updates += 1
            return False
        operations.remove(operation)
        if not operations:
            del self._inflight[block_id]
            self._backing_up_count -= 1
        if success:
            had_backup = bool(entry.residency & _LOWER_TIERS)
            entry.residency |= tier_value
            if not had_backup:
                self._backed_up_count += 1
        return True

    def drop_backup(
        self,
        block_id: int,
        generation: int,
        tier: KVResidencyTier,
    ) -> bool:
        """Forget a lower-tier copy if its generation is still current."""
        tier_value = self._validate_backup_tier(tier)
        entry = self.entries[block_id]
        if entry.generation != generation:
            self.stale_updates += 1
            return False
        had_backup = bool(entry.residency & _LOWER_TIERS)
        entry.residency &= ~tier_value
        if had_backup and not entry.residency & _LOWER_TIERS:
            self._backed_up_count -= 1
        return True

    @staticmethod
    def _validate_backup_tier(tier: KVResidencyTier) -> int:
        tier_value = int(tier)
        if tier_value not in (_TIER_CPU, _TIER_REMOTE):
            raise ValueError("backup tier must be CPU or REMOTE")
        return tier_value

    def _clear_transfer_state(
        self,
        block_id: int,
        entry: _BlockResidency,
    ) -> None:
        if self._inflight.pop(block_id, None) is not None:
            self._backing_up_count -= 1
        if entry.residency & _LOWER_TIERS:
            self._backed_up_count -= 1
        entry.residency = _TIER_NONE

    def _inflight_tiers(self, block_id: int) -> int:
        tiers = _TIER_NONE
        for tier, _ in self._inflight.get(block_id, ()):
            tiers |= tier
        return tiers

    @staticmethod
    def _matches_cached_generation(
        entry: _BlockResidency,
        generation: int,
    ) -> bool:
        return (
            entry.state != KVBlockState.UNTRACKED
            and entry.generation == generation
            and entry.hash_count > 0
        )

    def get_entry(self, block_id: int) -> KVResidencyEntry:
        if not 0 <= block_id < self.num_blocks:
            raise IndexError(block_id)
        entry = self.entries[block_id]
        backing_up_tiers = self._inflight_tiers(block_id)
        return KVResidencyEntry(
            block_id=block_id,
            generation=entry.generation,
            state=KVBlockState(entry.state),
            ref_count=entry.ref_count,
            request_ref_count=entry.request_ref_count,
            reuse_count=entry.reuse_count,
            peak_fanout=entry.peak_fanout,
            hash_count=entry.hash_count,
            residency=KVResidencyTier(entry.residency),
            backing_up=backing_up_tiers != _TIER_NONE,
            backing_up_tiers=KVResidencyTier(backing_up_tiers),
            last_access_ns=entry.last_access_ns,
        )

    def oldest(self, state: KVBlockState, limit: int) -> list[KVResidencyEntry]:
        """Return at most ``limit`` oldest entries without scanning other states."""
        return [
            self.get_entry(block_id) for block_id in self.oldest_block_ids(state, limit)
        ]

    def oldest_block_ids(self, state: KVBlockState, limit: int) -> list[int]:
        """Return oldest block IDs without constructing debug snapshots."""
        if state not in _IDLE_STATES:
            raise ValueError("only idle lifecycle segments have an ordered queue")
        if limit < 0:
            raise ValueError("limit must be non-negative")
        block_ids: list[int] = []
        block_id = self.heads[state]
        while block_id >= 0 and len(block_ids) < limit:
            block_ids.append(block_id)
            block_id = self.entries[block_id].next
        return block_ids

    def placement_metadata(self, block_id: int) -> tuple[int, bool, int, bool]:
        """Return generation, lower-copy, reuse, and in-flight state."""
        entry = self.entries[block_id]
        return (
            entry.generation,
            bool(entry.residency & _LOWER_TIERS),
            entry.reuse_count,
            block_id in self._inflight,
        )

    def can_reclaim(self, block_id: int, generation: int) -> bool:
        """Return whether a non-shared idle generation is reclaimable now."""
        entry = self.entries[block_id]
        return (
            entry.generation == generation
            and KVBlockState(entry.state)
            in (KVBlockState.WARM, KVBlockState.COOLING, KVBlockState.COLD)
            and block_id not in self._inflight
        )

    def backup_metadata(self, block_id: int) -> tuple[int, bool, bool, bool, bool]:
        """Return fields needed to validate a proactive backup candidate."""
        entry = self.entries[block_id]
        state = KVBlockState(entry.state)
        return (
            entry.generation,
            state in _IDLE_STATES,
            state in _SHARED_STATES,
            bool(entry.residency & _LOWER_TIERS),
            block_id in self._inflight,
        )

    def cache_occupancy(self) -> float:
        """Return the fraction of physical blocks holding cached data."""
        tracked_blocks = self.num_blocks - 1
        if tracked_blocks == 0:
            return 0.0
        return 1.0 - self.counts[KVBlockState.FREE] / tracked_blocks

    def state_count(self, state: KVBlockState) -> int:
        """Return the current number of blocks in one lifecycle state."""
        return self.counts[state]

    def snapshot(self) -> KVResidencyStats:
        """Build a constant-size metrics snapshot."""
        count = self.counts.__getitem__
        shared_blocks = sum(count(state) for state in _SHARED_STATES)
        return KVResidencyStats(
            epoch=self.epoch,
            tracked_blocks=self.num_blocks - count(KVBlockState.UNTRACKED),
            free_blocks=count(KVBlockState.FREE),
            unhashed_blocks=count(KVBlockState.UNHASHED),
            active_blocks=count(KVBlockState.ACTIVE),
            active_shared_blocks=count(KVBlockState.ACTIVE_SHARED),
            warm_blocks=count(KVBlockState.WARM),
            warm_shared_blocks=count(KVBlockState.WARM_SHARED),
            cooling_blocks=count(KVBlockState.COOLING),
            cooling_shared_blocks=count(KVBlockState.COOLING_SHARED),
            cold_blocks=count(KVBlockState.COLD),
            cold_shared_blocks=count(KVBlockState.COLD_SHARED),
            shared_blocks=shared_blocks,
            backing_up_blocks=self._backing_up_count,
            backed_up_blocks=self._backed_up_count,
            allocations=self.allocations,
            cache_hits=self.cache_hits,
            cache_insertions=self.cache_insertions,
            cache_removals=self.cache_removals,
            cache_evictions=self.cache_evictions,
            shared_cache_evictions=self.shared_cache_evictions,
            unbacked_cache_evictions=self.unbacked_cache_evictions,
            inflight_cache_evictions=self.inflight_cache_evictions,
            aging_transitions=self.aging_transitions,
            stale_updates=self.stale_updates,
        )

    def reset(self) -> None:
        """Reset all non-null blocks after a successful prefix-cache reset."""
        for block_id, entry in enumerate(self.entries):
            if block_id == self.null_block_id:
                continue
            self._next_generation(entry)
            entry.state = KVBlockState.FREE
            entry.ref_count = 0
            entry.request_ref_count = 0
            entry.reuse_count = 0
            entry.peak_fanout = 0
            entry.hash_count = 0
            entry.residency = _TIER_NONE
            entry.last_access_ns = self.now_ns
            entry.prev = -1
            entry.next = -1
            self._clear_restore_position(block_id)
        num_states = len(KVBlockState)
        self.heads = [-1] * num_states
        self.tails = [-1] * num_states
        self.counts = [0] * num_states
        self.counts[KVBlockState.UNTRACKED] = 1
        self.counts[KVBlockState.FREE] = self.num_blocks - 1
        self.aging_cursor = 0
        self._inflight.clear()
        self._backing_up_count = 0
        self._backed_up_count = 0
        assert self._backing_up_count == 0
        assert self._backed_up_count == 0

    def check_consistency(self) -> None:
        """Expensive invariant check intended for tests and diagnostics."""
        seen: set[int] = set()
        for state in _IDLE_STATES:
            block_id = self.heads[state]
            prev_id = -1
            count = 0
            while block_id >= 0:
                entry = self.entries[block_id]
                assert block_id not in seen
                assert entry.state == state
                assert entry.prev == prev_id
                seen.add(block_id)
                prev_id = block_id
                block_id = entry.next
                count += 1
            assert self.tails[state] == prev_id
            assert self.counts[state] == count
        actual_counts = [0] * len(KVBlockState)
        for entry in self.entries:
            actual_counts[entry.state] += 1
            assert 0 <= entry.request_ref_count <= entry.ref_count
        assert self.counts == actual_counts
        assert len(seen) == sum(self.counts[state] for state in _IDLE_STATES)
        assert self._backing_up_count == len(self._inflight)
        assert self._backed_up_count == sum(
            bool(entry.residency & _LOWER_TIERS) for entry in self.entries
        )


def create_kv_residency_index(block_pool: "BlockPool") -> KVResidencyIndex | None:
    """Build the optional observer without leaking its config into BlockPool."""
    if not (
        envs.VLLM_AGENTRIX_KV_RESIDENCY_SHADOW
        or envs.VLLM_AGENTRIX_KV_PLACEMENT_SHADOW
        or envs.VLLM_AGENTRIX_KV_PLACEMENT_ACTIVE
        or envs.VLLM_AGENTRIX_KV_PROACTIVE_BACKUP
    ):
        return None
    return KVResidencyIndex(
        block_pool.num_gpu_blocks,
        null_block_id=block_pool.null_block.block_id,
        warm_seconds=envs.VLLM_AGENTRIX_KV_WARM_SECONDS,
        cooling_seconds=envs.VLLM_AGENTRIX_KV_COOLING_SECONDS,
        shared_reuse_threshold=envs.VLLM_AGENTRIX_KV_SHARED_REUSE_THRESHOLD,
        aging_budget=envs.VLLM_AGENTRIX_KV_AGING_BUDGET,
    )
