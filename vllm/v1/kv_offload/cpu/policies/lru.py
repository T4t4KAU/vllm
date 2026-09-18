# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class LRUCachePolicy(CachePolicy):
    """
    LRU Caching policy that keeps a dedicated evictable list for fast eviction.
    A use is indicated by,
     - First time the key is added (store).
     - Load job completion
     - touch
    """

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        # Blocks with ref_cnt 0 (not participating in any loads/stores) ordered in LRU
        self.evictable_blocks: OrderedDict[OffloadKey, None] = OrderedDict()
        self.blocks: dict[OffloadKey, BlockStatus] = {}

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        if block.ref_cnt == 0:
            self.evictable_blocks[key] = None

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self.evictable_blocks.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        for key in reversed(list(keys)):
            if key in self.evictable_blocks:
                self.evictable_blocks.move_to_end(key)
            # active blocks are untouched as they are non-evictable now. They
            # will eventually reach the end of evictable_blocks when they finish.

    @override
    def clear(self) -> None:
        self.evictable_blocks.clear()
        self.blocks.clear()

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for key, _ in self.evictable_blocks.items():
            if key in protected:
                continue

            block = self.blocks[key]
            assert block.ref_cnt == 0
            candidates.append((key, block))
            if len(candidates) == n:
                break

        if len(candidates) < n:
            return None
        for key, _ in candidates:
            del self.evictable_blocks[key]
            del self.blocks[key]
        return candidates

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        # blocks can become evictable when,
        # store completes - i.e. ref_cnt -1 -> 0 # not in evictable list
        # all loads complete - i.e ref_cnt 1 -> 0  # not in evictable list
        self.evictable_blocks[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        # key must have been in the evictable list.
        del self.evictable_blocks[key]


@dataclass
class _SessionLease:
    active: set[str] = field(default_factory=set)
    deadline: float = 0.0


class SessionLRUCachePolicy(LRUCachePolicy):
    """Experimental, bounded soft retention across turns of an Agent task tree.

    Optional ``kv_transfer_params.agentrix_session.root_session_id`` groups
    related sessions. Missing hints fall back to the request's session_id.
    Leases only influence eviction order: allocation may reclaim leased blocks.
    """

    def __init__(
        self,
        cache_capacity: int,
        retention_seconds: float = 120.0,
        protected_fraction: float = 0.5,
        max_sessions: int = 256,
    ):
        super().__init__(cache_capacity)
        if not isfinite(retention_seconds) or retention_seconds <= 0:
            raise ValueError("retention_seconds must be finite and positive")
        if not isfinite(protected_fraction) or not 0 <= protected_fraction <= 1:
            raise ValueError("protected_fraction must be between zero and one")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.retention_seconds = retention_seconds
        self.max_retained_blocks = int(cache_capacity * protected_fraction)
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, _SessionLease] = OrderedDict()
        self._retained: OrderedDict[OffloadKey, list[str]] = OrderedDict()

    @staticmethod
    def _session_id(ctx: ReqContext) -> str | None:
        hints = (ctx.kv_transfer_params or {}).get("agentrix_session")
        root = hints.get("root_session_id") if isinstance(hints, dict) else None
        session = root if isinstance(root, str) and root else ctx.session_id
        return session if isinstance(session, str) and session else None

    @override
    def on_new_request(self, req_context: ReqContext) -> None:
        sid = self._session_id(req_context)
        if sid is None or not self.max_retained_blocks:
            return
        if sid not in self._sessions:
            if len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)
            self._sessions[sid] = _SessionLease()
        self._sessions.move_to_end(sid)
        self._sessions[sid].active.add(req_context.req_id)

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        sid = self._session_id(req_context)
        session = self._sessions.get(sid) if sid is not None else None
        if session is not None:
            session.active.discard(req_context.req_id)
            session.deadline = monotonic() + self.retention_seconds

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        sid = self._session_id(req_context)
        if sid is None or sid not in self._sessions:
            super().touch(keys, req_context)
            return
        # Keep earlier prefix blocks when the retention budget is exceeded.
        for key in reversed(list(keys)):
            if key in self.evictable_blocks:
                self.evictable_blocks.move_to_end(key)
            if key not in self.blocks:
                continue
            owners = self._retained.get(key)
            if owners is None:
                self._retained[key] = [sid]
            elif owners[-1] != sid:
                if sid in owners:
                    owners.remove(sid)
                owners.append(sid)
                # Forgetting an old owner drops priority, never a KV reference.
                del owners[:-4]
            self._retained.move_to_end(key)
            if len(self._retained) > self.max_retained_blocks:
                self._retained.popitem(last=False)

    @override
    def remove(self, key: OffloadKey) -> None:
        super().remove(key)
        self._retained.pop(key, None)

    @override
    def clear(self) -> None:
        super().clear()
        self._sessions.clear()
        self._retained.clear()

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        if not self._retained:
            return super().evict(n, protected)
        now = monotonic()
        live_sessions = {
            sid
            for sid, session in self._sessions.items()
            if session.active or session.deadline > now
        }
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        deferred: list[tuple[OffloadKey, BlockStatus]] = []
        for key in self.evictable_blocks:
            if key in protected:
                continue
            entry = (key, self.blocks[key])
            owners = self._retained.get(key)
            leased = owners and not live_sessions.isdisjoint(owners)
            if leased:
                deferred.append(entry)
            else:
                candidates.append(entry)
                if len(candidates) == n:
                    break
        candidates.extend(deferred[: n - len(candidates)])
        if len(candidates) < n:
            return None
        for key, _ in candidates:
            self.remove(key)
        return candidates
