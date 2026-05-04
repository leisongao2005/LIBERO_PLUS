# Hierarchical sim ↔ wrapper documentation

**Ground truth for new work:** [.cursor/plans/hierarchical_sim_wrapper_refactor_599aa945.plan.md](../../../../.cursor/plans/hierarchical_sim_wrapper_refactor_599aa945.plan.md) (from repo root: `.cursor/plans/…`). Follow that plan for the Markovian sim layer, new `info` contract, and `HierarchicalRewardWrapper` behavior unless a maintainer explicitly updates the plan.

**Phase 0 sim `step` contract (canonical code):** [`libero/libero/bddlsim_interface.py`](../../bddlsim_interface.py). The path [`_interface.py`](./_interface.py) in this package is a thin re-export for import convenience.

**Legacy reference only (describes pre-refactor `BDDLBaseDomain` behavior):** [`../SUBTASK_STEP_INFO.md`](../SUBTASK_STEP_INFO.md), [`../SUBTASK_CONTROL_FLOW.md`](../SUBTASK_CONTROL_FLOW.md), [`../PREDICATE_LEVEL_EVALUATION.md`](../PREDICATE_LEVEL_EVALUATION.md). Use them to understand the old monolithic pipeline, not as the target contract after the refactor.

**Related project markdown (not the refactor spec):** repository [`README.md`](../../../../README.md); [`scripts/COLLECT_DEMONSTRATION_LIBERO_PLUS.md`](../../../../scripts/COLLECT_DEMONSTRATION_LIBERO_PLUS.md) (demo collection, BDDL paths); [`scripts/leison/BENCHMARK_RESULTS.md`](../../../../scripts/leison/BENCHMARK_RESULTS.md) (predicate timing, mentions `subtask_reward`).
