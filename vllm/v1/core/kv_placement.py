# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded planning and application for KV cache placement."""

from dataclasses import dataclass, field
from enum import IntEnum
from operator import attrgetter
from typing import TYPE_CHECKING

from vllm import envs
from vllm.v1.core.kv_residency import (
    KVBlockState,
    KVResidencyIndex,
)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool


class KVPlacementAction(IntEnum):
    """Action proposed for an idle GPU block."""

    BACKUP_TO_CPU = 1
    RELEASE_GPU = 2
    DISCARD = 3
    DISCARD_UNBACKED = 4


@dataclass(frozen=True, slots=True)
class KVPlacementDecision:
    """One generation-safe placement decision."""

    block_id: int
    generation: int
    state: KVBlockState
    action: KVPlacementAction


@dataclass(frozen=True, slots=True)
class KVPlacementPlan:
    """Bounded plan for one allocation-pressure event."""

    epoch: int
    requested_blocks: int
    scanned_blocks: int
    decisions: tuple[KVPlacementDecision, ...]
    deferred_blocks: int
    excluded_blocks: int
    protected_shared_blocks: int
    protected_shortfall_blocks: int
    unresolved_blocks: int
    scan_budget_exhausted: bool


@dataclass(frozen=True, slots=True)
class KVPlacementStats:
    """Cumulative placement-planner statistics."""

    pressure_events: int = 0
    requested_blocks: int = 0
    scanned_blocks: int = 0
    planned_blocks: int = 0
    backup_to_cpu_blocks: int = 0
    release_gpu_blocks: int = 0
    discarded_blocks: int = 0
    unbacked_discarded_blocks: int = 0
    deferred_blocks: int = 0
    excluded_blocks: int = 0
    protected_shortfall_blocks: int = 0
    unresolved_blocks: int = 0
    budget_exhaustions: int = 0
    applied_plans: int = 0
    applied_blocks: int = 0
    active_shortfall_blocks: int = 0
    apply_failures: int = 0


@dataclass(slots=True)
class _KVPlacementCounters:
    pressure_events: int = 0
    requested_blocks: int = 0
    scanned_blocks: int = 0
    planned_blocks: int = 0
    backup_to_cpu_blocks: int = 0
    release_gpu_blocks: int = 0
    discarded_blocks: int = 0
    unbacked_discarded_blocks: int = 0
    deferred_blocks: int = 0
    excluded_blocks: int = 0
    protected_shortfall_blocks: int = 0
    unresolved_blocks: int = 0
    budget_exhaustions: int = 0
    applied_plans: int = 0
    applied_blocks: int = 0
    active_shortfall_blocks: int = 0
    apply_failures: int = 0


@dataclass(slots=True)
class _KVPlacementCandidate:
    """Reusable mutable candidate for the active scheduler path."""

    block_id: int = 0
    generation: int = 0
    state: KVBlockState = KVBlockState.UNTRACKED
    action: KVPlacementAction = KVPlacementAction.DISCARD
    priority: int = 0


@dataclass(slots=True)
class _KVPlacementWorkspace:
    """Reusable scan result, materialized only for diagnostics."""

    epoch: int = 0
    requested_blocks: int = 0
    scanned_blocks: int = 0
    decision_count: int = 0
    deferred_blocks: int = 0
    excluded_blocks: int = 0
    protected_shared_blocks: int = 0
    protected_shortfall_blocks: int = 0
    unresolved_blocks: int = 0
    scan_budget_exhausted: bool = False
    has_plan: bool = False
    candidates: list[_KVPlacementCandidate] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)


_VICTIM_STATES = (
    KVBlockState.COLD,
    KVBlockState.COOLING,
    KVBlockState.WARM,
)
_PROTECTED_STATES = (
    KVBlockState.COLD_SHARED,
    KVBlockState.COOLING_SHARED,
    KVBlockState.WARM_SHARED,
)
_CANDIDATE_PRIORITY = attrgetter("priority")
_EMPTY_BLOCK_IDS: frozenset[int] = frozenset()


class KVPlacementPlanner:
    """Propose placement changes without modifying cache state.

    Shared prefixes are a hard exclusion. Other idle blocks are considered by
    temperature and then age. Shadow scans have a strict per-step bound;
    active scans may additionally inspect one entry per requested allocation
    block so large allocations can make progress without a cache-wide scan.
    """

    __slots__ = (
        "scan_budget",
        "active",
        "_budget_epoch",
        "_scanned_this_epoch",
        "_last_plan",
        "_counters",
        "_workspace",
    )

    def __init__(self, scan_budget: int = 64, *, active: bool = False) -> None:
        if scan_budget <= 0:
            raise ValueError("scan_budget must be positive")
        self.scan_budget = scan_budget
        self.active = active
        self._budget_epoch = -1
        self._scanned_this_epoch = 0
        self._last_plan: KVPlacementPlan | None = None
        self._counters = _KVPlacementCounters()
        self._workspace = _KVPlacementWorkspace()

    @property
    def last_plan(self) -> KVPlacementPlan | None:
        """Return an immutable snapshot of the most recent planning event."""
        workspace = self._workspace
        if self._last_plan is None and workspace.has_plan:
            self._last_plan = self._materialize_plan(workspace)
        return self._last_plan

    def plan(
        self,
        index: KVResidencyIndex,
        requested_blocks: int,
        *,
        excluded_block_ids: set[int] | None = None,
    ) -> KVPlacementPlan:
        """Plan enough non-shared victims to satisfy one allocation."""
        workspace = self._scan(
            index,
            requested_blocks,
            excluded_block_ids=excluded_block_ids,
        )
        self._record(workspace)
        plan = self._materialize_plan(workspace)
        self._last_plan = plan
        return plan

    def plan_and_apply(
        self,
        index: KVResidencyIndex,
        requested_blocks: int,
        block_pool: "BlockPool",
        *,
        excluded_block_ids: set[int] | None = None,
    ) -> bool:
        """Plan and atomically apply an active placement event.

        This allocation hot path reuses mutable candidates. It materializes an
        immutable ``KVPlacementPlan`` only when diagnostics read ``last_plan``.

        Args:
            index: Residency metadata used to select and validate victims.
            requested_blocks: Number of cached GPU blocks to reclaim.
            block_pool: Pool whose free queue will consume the selected blocks.
            excluded_block_ids: Cache-hit blocks acquired by this allocation.

        Returns:
            True when the full request was applied atomically.

        Raises:
            RuntimeError: If called while active placement is disabled.
        """
        if not self.active:
            raise RuntimeError("plan_and_apply requires active placement")
        if requested_blocks < 0:
            raise ValueError("requested_blocks must be non-negative")
        if requested_blocks == 0:
            return True
        workspace = self._scan(
            index,
            requested_blocks,
            excluded_block_ids=excluded_block_ids,
        )
        self._record(workspace)
        return self._apply_workspace(workspace, index, block_pool)

    def plan_and_apply_for_allocation(
        self,
        index: KVResidencyIndex,
        num_blocks_to_allocate: int,
        block_pool: "BlockPool",
        *,
        excluded_block_ids: set[int] | None = None,
    ) -> bool:
        """Apply active placement for the cached portion of an allocation.

        Args:
            index: Residency metadata used to select and validate victims.
            num_blocks_to_allocate: Total blocks reported by the coordinator.
            block_pool: Pool whose free queue will consume the selected blocks.
            excluded_block_ids: Cache-hit blocks acquired by this allocation.

        Returns:
            True when no recycling is needed or all victims were applied.
        """
        requested_blocks = self._recycled_block_count(
            index,
            num_blocks_to_allocate,
            excluded_block_ids,
        )
        if requested_blocks == 0:
            return True
        return self.plan_and_apply(
            index,
            requested_blocks,
            block_pool,
            excluded_block_ids=excluded_block_ids,
        )

    def _scan(
        self,
        index: KVResidencyIndex,
        requested_blocks: int,
        *,
        excluded_block_ids: set[int] | None,
    ) -> _KVPlacementWorkspace:
        if requested_blocks < 0:
            raise ValueError("requested_blocks must be non-negative")
        if index.epoch != self._budget_epoch:
            self._budget_epoch = index.epoch
            self._scanned_this_epoch = 0

        excluded_ids = (
            excluded_block_ids if excluded_block_ids is not None else _EMPTY_BLOCK_IDS
        )
        workspace = self._workspace
        candidates = workspace.candidates
        candidate_count = 0
        scanned_blocks = 0
        deferred_blocks = 0
        excluded_blocks = 0
        remaining_step_budget = max(
            0,
            self.scan_budget - self._scanned_this_epoch,
        )
        available_budget = remaining_step_budget
        if self.active:
            available_budget = max(
                available_budget,
                requested_blocks + len(excluded_ids),
            )
        unprotected_blocks = sum(index.state_count(state) for state in _VICTIM_STATES)

        for state in _VICTIM_STATES:
            remaining_budget = available_budget - scanned_blocks
            if remaining_budget == 0 or (
                not self.active and candidate_count == requested_blocks
            ):
                break
            for block_id in index.oldest_block_ids(state, remaining_budget):
                scanned_blocks += 1
                if block_id in excluded_ids:
                    excluded_blocks += 1
                    continue
                generation, has_lower_copy, reuse_count, backing_up = (
                    index.placement_metadata(block_id)
                )
                if backing_up:
                    deferred_blocks += 1
                    continue

                if has_lower_copy:
                    action = KVPlacementAction.RELEASE_GPU
                    priority = 0
                elif state == KVBlockState.COLD and reuse_count == 0:
                    action = KVPlacementAction.DISCARD
                    priority = 1
                elif self.active:
                    action = KVPlacementAction.DISCARD_UNBACKED
                    priority = 2
                else:
                    action = KVPlacementAction.BACKUP_TO_CPU
                    priority = 2

                if candidate_count == len(candidates):
                    candidates.append(_KVPlacementCandidate())
                candidate = candidates[candidate_count]
                candidate.block_id = block_id
                candidate.generation = generation
                candidate.state = state
                candidate.action = action
                candidate.priority = priority
                candidate_count += 1
                if not self.active and candidate_count == requested_blocks:
                    break

        self._scanned_this_epoch += scanned_blocks

        if candidate_count < len(candidates):
            del candidates[candidate_count:]
        if self.active:
            candidates.sort(key=_CANDIDATE_PRIORITY)
        decision_count = min(candidate_count, requested_blocks)

        unresolved_blocks = requested_blocks - decision_count
        scan_budget_exhausted = (
            unresolved_blocks > 0
            and scanned_blocks == available_budget
            and scanned_blocks < unprotected_blocks
        )
        protected_shared_blocks = sum(
            index.state_count(state) for state in _PROTECTED_STATES
        )
        protected_shortfall_blocks = 0
        if unresolved_blocks > 0 and not scan_budget_exhausted:
            shortfall_after_deferred = max(0, unresolved_blocks - deferred_blocks)
            protected_shortfall_blocks = min(
                shortfall_after_deferred,
                protected_shared_blocks,
            )

        workspace.epoch = index.epoch
        workspace.requested_blocks = requested_blocks
        workspace.scanned_blocks = scanned_blocks
        workspace.decision_count = decision_count
        workspace.deferred_blocks = deferred_blocks
        workspace.excluded_blocks = excluded_blocks
        workspace.protected_shared_blocks = protected_shared_blocks
        workspace.protected_shortfall_blocks = protected_shortfall_blocks
        workspace.unresolved_blocks = unresolved_blocks
        workspace.scan_budget_exhausted = scan_budget_exhausted
        workspace.has_plan = True
        self._last_plan = None
        return workspace

    def plan_for_allocation(
        self,
        index: KVResidencyIndex,
        num_blocks_to_allocate: int,
        *,
        excluded_block_ids: set[int] | None = None,
    ) -> KVPlacementPlan | None:
        """Plan only the allocation portion that must recycle cached blocks."""
        requested_blocks = self._recycled_block_count(
            index,
            num_blocks_to_allocate,
            excluded_block_ids,
        )
        if requested_blocks == 0:
            return None
        return self.plan(
            index,
            requested_blocks,
            excluded_block_ids=excluded_block_ids,
        )

    def apply(
        self,
        plan: KVPlacementPlan,
        index: KVResidencyIndex,
        block_pool: "BlockPool",
    ) -> bool:
        """Atomically prioritize an active plan in the block pool.

        Args:
            plan: Generation-safe placement decisions for one allocation.
            index: Residency index used to revalidate each decision.
            block_pool: Pool whose free queue will consume the decisions.

        Returns:
            True for shadow plans and fully applied active plans.
        """
        if not self.active:
            return True
        if plan.unresolved_blocks > 0:
            self._counters.active_shortfall_blocks += plan.unresolved_blocks
            return False
        if not all(
            index.can_reclaim(decision.block_id, decision.generation)
            for decision in plan.decisions
        ):
            self._counters.apply_failures += 1
            return False
        block_ids = [decision.block_id for decision in plan.decisions]
        if not block_pool.prioritize_eviction_blocks(block_ids):
            self._counters.apply_failures += 1
            return False
        self._counters.applied_plans += 1
        self._counters.applied_blocks += len(block_ids)
        return True

    def _apply_workspace(
        self,
        workspace: _KVPlacementWorkspace,
        index: KVResidencyIndex,
        block_pool: "BlockPool",
    ) -> bool:
        if workspace.unresolved_blocks > 0:
            self._counters.active_shortfall_blocks += workspace.unresolved_blocks
            return False

        block_ids = workspace.block_ids
        for candidate_index in range(workspace.decision_count):
            candidate = workspace.candidates[candidate_index]
            if not index.can_reclaim(candidate.block_id, candidate.generation):
                self._counters.apply_failures += 1
                return False
            if candidate_index == len(block_ids):
                block_ids.append(candidate.block_id)
            else:
                block_ids[candidate_index] = candidate.block_id
        if workspace.decision_count < len(block_ids):
            del block_ids[workspace.decision_count :]

        if not block_pool.prioritize_eviction_blocks(block_ids):
            self._counters.apply_failures += 1
            return False
        self._counters.applied_plans += 1
        self._counters.applied_blocks += workspace.decision_count
        return True

    @staticmethod
    def _recycled_block_count(
        index: KVResidencyIndex,
        num_blocks_to_allocate: int,
        excluded_block_ids: set[int] | None,
    ) -> int:
        if num_blocks_to_allocate < 0:
            raise ValueError("num_blocks_to_allocate must be non-negative")
        uncached_free_blocks = index.state_count(KVBlockState.FREE)
        num_excluded_blocks = len(excluded_block_ids or ())
        return max(
            0,
            num_blocks_to_allocate - uncached_free_blocks - num_excluded_blocks,
        )

    @staticmethod
    def _materialize_plan(workspace: _KVPlacementWorkspace) -> KVPlacementPlan:
        decisions = tuple(
            KVPlacementDecision(
                block_id=candidate.block_id,
                generation=candidate.generation,
                state=candidate.state,
                action=candidate.action,
            )
            for candidate in workspace.candidates[: workspace.decision_count]
        )
        return KVPlacementPlan(
            epoch=workspace.epoch,
            requested_blocks=workspace.requested_blocks,
            scanned_blocks=workspace.scanned_blocks,
            decisions=decisions,
            deferred_blocks=workspace.deferred_blocks,
            excluded_blocks=workspace.excluded_blocks,
            protected_shared_blocks=workspace.protected_shared_blocks,
            protected_shortfall_blocks=workspace.protected_shortfall_blocks,
            unresolved_blocks=workspace.unresolved_blocks,
            scan_budget_exhausted=workspace.scan_budget_exhausted,
        )

    def _record(self, workspace: _KVPlacementWorkspace) -> None:
        counters = self._counters
        counters.pressure_events += 1
        counters.requested_blocks += workspace.requested_blocks
        counters.scanned_blocks += workspace.scanned_blocks
        counters.planned_blocks += workspace.decision_count
        counters.deferred_blocks += workspace.deferred_blocks
        counters.excluded_blocks += workspace.excluded_blocks
        counters.protected_shortfall_blocks += workspace.protected_shortfall_blocks
        counters.unresolved_blocks += workspace.unresolved_blocks
        counters.budget_exhaustions += workspace.scan_budget_exhausted
        for candidate_index in range(workspace.decision_count):
            action = workspace.candidates[candidate_index].action
            if action == KVPlacementAction.BACKUP_TO_CPU:
                counters.backup_to_cpu_blocks += 1
            elif action == KVPlacementAction.RELEASE_GPU:
                counters.release_gpu_blocks += 1
            else:
                counters.discarded_blocks += 1
                counters.unbacked_discarded_blocks += (
                    action == KVPlacementAction.DISCARD_UNBACKED
                )

    def snapshot(self) -> KVPlacementStats:
        """Return constant-size cumulative planner statistics."""
        counters = self._counters
        return KVPlacementStats(
            pressure_events=counters.pressure_events,
            requested_blocks=counters.requested_blocks,
            scanned_blocks=counters.scanned_blocks,
            planned_blocks=counters.planned_blocks,
            backup_to_cpu_blocks=counters.backup_to_cpu_blocks,
            release_gpu_blocks=counters.release_gpu_blocks,
            discarded_blocks=counters.discarded_blocks,
            unbacked_discarded_blocks=counters.unbacked_discarded_blocks,
            deferred_blocks=counters.deferred_blocks,
            excluded_blocks=counters.excluded_blocks,
            protected_shortfall_blocks=counters.protected_shortfall_blocks,
            unresolved_blocks=counters.unresolved_blocks,
            budget_exhaustions=counters.budget_exhaustions,
            applied_plans=counters.applied_plans,
            applied_blocks=counters.applied_blocks,
            active_shortfall_blocks=counters.active_shortfall_blocks,
            apply_failures=counters.apply_failures,
        )


def create_kv_placement_planner() -> KVPlacementPlanner | None:
    """Create the optional placement planner from environment configuration."""
    active = envs.VLLM_AGENTRIX_KV_PLACEMENT_ACTIVE
    if not (envs.VLLM_AGENTRIX_KV_PLACEMENT_SHADOW or active):
        return None
    return KVPlacementPlanner(
        envs.VLLM_AGENTRIX_KV_PLACEMENT_SCAN_BUDGET,
        active=active,
    )
