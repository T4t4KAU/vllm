# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observer contract for KV cache block-pool lifecycle transitions."""

from typing import Protocol


class KVCacheObserver(Protocol):
    """Optional, policy-free observer attached to a GPU block pool.

    Implementations must keep callbacks non-blocking. In particular, callbacks
    must not perform device copies, RPCs, or cache-wide scans.
    """

    def on_allocated(self, block_id: int, ref_count: int) -> int: ...

    def on_cache_inserted(self, block_id: int) -> None: ...

    def on_cache_hit(self, block_id: int, ref_count: int) -> None: ...

    def on_pinned(self, block_id: int, ref_count: int) -> None: ...

    def on_released(self, block_id: int, ref_count: int) -> None: ...

    def on_unpinned(self, block_id: int, ref_count: int) -> None: ...

    def on_cache_removed(
        self,
        block_id: int,
        num_hashes: int,
        *,
        evicted: bool = False,
    ) -> None: ...

    def on_step(self) -> None: ...

    def reset(self) -> None: ...
