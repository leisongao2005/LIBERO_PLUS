# Predicate benchmark results

**Run configuration:** `steps=100`, `check_loops=500`

## Summary table


| BDDL                                    | Predicate                | `n_goal_atoms` | Tier A mean (ms/step) | Tier A std | Tier B mean (ms/check) | Tier B std |
| --------------------------------------- | ------------------------ | -------------- | --------------------- | ---------- | ---------------------- | ---------- |
| `close_kitchen_scene6.bddl`             | close                    | 1              | 6.6242                | 0.2860     | 0.0073                 | 0.0007     |
| `default_grasp_kitchen_scene6.bddl`     | defaultgrasppredicate    | 1              | 6.2776                | 0.0364     | 0.1481                 | 0.0239     |
| `exactin_kitchen_scene6.bddl`           | exactin                  | 1              | 6.1070                | 0.0603     | 0.0105                 | 0.0001     |
| `grasp_kitchen_scene6.bddl`             | grasp                    | 1              | 6.2394                | 0.1108     | 0.1287                 | 0.0005     |
| `horizontal_grasp_kitchen_scene6.bddl`  | horizontalgrasppredicate | 1              | 6.1953                | 0.0123     | 0.1298                 | 0.0003     |
| `in_kitchen_scene6.bddl`                | in                       | 1              | 5.9662                | 0.0290     | 0.0068                 | 0.0001     |
| `localized_neareef_kitchen_scene6.bddl` | localizedneareef         | 1              | 6.2846                | 0.1945     | 0.0044                 | 0.0001     |
| `near_mugs_kitchen_scene6.bddl`         | near                     | 1              | 6.0227                | 0.0251     | 0.0050                 | 0.0000     |
| `neareef_kitchen_scene6.bddl`           | neareef                  | 1              | 6.0481                | 0.0200     | 0.0044                 | 0.0001     |
| `on_kitchen_scene8.bddl`                | on                       | 1              | 6.3556                | 0.0367     | 0.0069                 | 0.0002     |
| `turnon_kitchen_scene8.bddl`            | turnon                   | 1              | 6.4780                | 0.0867     | 0.0069                 | 0.0002     |


## Methodology

- **Tier A:** full environment step (ms per step).
- **Tier B:** `_check_success` only (ms per check).
- With `subtask_reward=False`, expect roughly `2 × n_goal_atoms` predicate passes per step.

