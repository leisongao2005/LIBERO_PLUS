#!/usr/bin/env python3
"""
Predicate micro-benchmarks for LIBERO BDDL goals.

Goal multiplicity (not confounded by the *original* multi-predicate task):
  ``generate`` copies a full task BDDL for :init / objects / regions, but replaces
  the entire ``(:goal ...)`` block with ``(And (<single atom>))``. Parsed
  ``goal_state`` should therefore have **one** conjunct per benchmark file—the
  predicate you are measuring—not the original task's multi-atom goal.

What *is* confounded (especially Tier A):
  ``env.step()`` includes MuJoCo integration, contacts, observations, rendering
  setup, etc. Tier A is **not** a pure predicate timer; use Tier B
  (``--check-loops``) for time dominated by repeated ``_check_success()`` /
  predicate evaluation on the current state. Even Tier B still runs on the
  full scene; it isolates *success checking* from physics stepping.

With subtask_reward=False, each env.step() triggers _check_success() twice per
control step (reward() and done in BDDLBaseDomain.step), so each goal conjunct
runs about 2 * len(goal_state) times per step (2x when len(goal_state)==1).

Usage (from repo root, with libero installed / PYTHONPATH set):
  python scripts/leison/benchmark_predicates.py generate [--config PATH] [--clean]
  python scripts/leison/benchmark_predicates.py run [--bddl-dir DIR] [--steps 100] ...

See predicate_benchmark_targets.yaml for the editable target list.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import sys
import time
from typing import Any, List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _resolve_bddl_files_root() -> str:
    repo_bddl = os.path.join(_REPO_ROOT, "libero", "libero", "bddl_files")
    if os.path.isdir(repo_bddl):
        return repo_bddl
    try:
        from libero.libero.utils import get_libero_path

        return get_libero_path("bddl_files")
    except Exception:
        return repo_bddl


def default_benchmark_bddl_dir() -> str:
    return os.path.join(_resolve_bddl_files_root(), "predicate_benchmark_temp")


_HEAD_CAPS = {
    "in": "In",
    "on": "On",
    "close": "Close",
    "open": "Open",
    "turnon": "Turnon",
    "turnoff": "Turnoff",
    "grasp": "Grasp",
    "neareef": "NearEEF",
    "near": "Near",
    "exactin": "ExactIn",
    "defaultgrasppredicate": "DefaultGraspPredicate",
    "horizontalgrasppredicate": "HorizontalGraspPredicate",
    "localizedneareef": "LocalizedNearEEF",
}


def _format_goal_token(tok: Any) -> str:
    if isinstance(tok, float):
        return str(int(tok)) if tok == int(tok) else repr(tok)
    if isinstance(tok, int):
        return str(tok)
    return str(tok)


def _format_result_row(r: dict[str, Any]) -> str:
    """Human-readable summary so Tier A vs Tier B ms is obvious (not a one-line dict)."""
    lines = [
        "",
        "  " + "=" * 60,
        f"  bddl:            {r.get('bddl', '')}",
        f"  predicates:      {r.get('predicates', '')}",
        f"  n_goal_atoms:    {r.get('n_goal_atoms', '')}",
        "  --- timing ---",
        f"  Tier A (ms/step):   mean={r.get('tier_a_mean_ms_per_step', '')!s}  "
        f"std={r.get('tier_a_std_ms', '')!s}",
    ]
    tb_mean = r.get("tier_b_mean_ms_per_check", "")
    if tb_mean not in ("", None):
        lines.append(
            f"  Tier B (ms/check):  mean={r.get('tier_b_mean_ms_per_check', '')!s}  "
            f"std={r.get('tier_b_std_ms', '')!s}"
        )
    else:
        lines.append("  Tier B (ms/check):  (skipped; use --check-loops > 0)")
    note = r.get("note") or ""
    if note:
        lines.append(f"  note: {note}")
    lines.append(
        f"  (steps={r.get('steps', '')}, check_loops={r.get('check_loops', '')})"
    )
    return "\n".join(lines)


def atom_list_to_s_expr(atom: List[Any]) -> str:
    head = str(atom[0]).lower()
    head_out = _HEAD_CAPS.get(head, str(atom[0]))
    inner = " ".join(_format_goal_token(x) for x in atom[1:])
    return f"({head_out}{(' ' + inner) if inner else ''})"


def _coerce_match_predicate(value: Any) -> Optional[str]:
    """YAML 1.1 parses unquoted *on* as bool True; normalize to the predicate name *on*."""
    if value is None:
        return None
    if value is True:
        return "on"
    if isinstance(value, bool):
        raise ValueError(
            "match_predicate was parsed as a boolean (YAML treats unquoted "
            "on/off/yes/no as booleans). Quote it, e.g. match_predicate: 'on'"
        )
    s = str(value).strip()
    return s or None


def find_goal_atom(goal_state: List, match_predicate: str) -> Optional[List[Any]]:
    p = match_predicate.lower()
    for atom in goal_state:
        if atom and str(atom[0]).lower() == p:
            return list(atom)
    return None


def _scan_balanced_paren(text: str, open_index: int) -> int:
    depth = 0
    k = open_index
    while k < len(text):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1
    raise ValueError("Unbalanced parentheses in BDDL")


def replace_goal_block(bddl_text: str, inner_and: str) -> str:
    marker = "(:goal"
    i = bddl_text.find(marker)
    if i == -1:
        raise ValueError("No (:goal ...) block found")
    # i points at '(' of '( :goal <body> )'. Scan from i so we remove the whole
    # form including its closing ')'. Scanning from the inner '(And ...)' left
    # an extra ')' and broke BDDL ("Missing open parenthesis").
    end = _scan_balanced_paren(bddl_text, i)
    new_block = f"  (:goal\n    {inner_and}\n  )"
    return bddl_text[:i] + new_block + bddl_text[end:]


def strip_subtask_rewards_block(bddl_text: str) -> str:
    key = "(:subtask_rewards"
    while True:
        i = bddl_text.find(key)
        if i == -1:
            return bddl_text
        j = bddl_text.find("(", i)
        end = _scan_balanced_paren(bddl_text, j)
        bddl_text = (bddl_text[:i].rstrip() + "\n\n" + bddl_text[end:].lstrip("\n")).strip() + "\n"


def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def _load_bddl_utils_standalone():
    """Load bddl_utils without importing libero.envs (avoids robosuite for generate-only)."""
    import importlib.util

    path = os.path.join(_REPO_ROOT, "libero", "libero", "envs", "bddl_utils.py")
    spec = importlib.util.spec_from_file_location("_libero_bddl_utils_bench", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def cmd_generate(args: argparse.Namespace) -> None:
    import yaml

    bddl_utils = _load_bddl_utils_standalone()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    suite = cfg.get("suite", "libero_10")
    targets = cfg.get("targets") or []
    bddl_root = _resolve_bddl_files_root()
    out_dir = os.path.abspath(args.bddl_dir or default_benchmark_bddl_dir())
    suite_dir = os.path.join(bddl_root, suite)

    if args.clean and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    written = 0
    for t in targets:
        if not t.get("enabled", True):
            continue
        tid = t.get("id")
        source_task = t.get("source_task")
        if not tid or not source_task:
            raise ValueError(f"target missing id or source_task: {t}")
        src_path = os.path.join(suite_dir, f"{source_task}.bddl")
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Missing source BDDL: {src_path}")

        parsed = bddl_utils.robosuite_parse_problem(src_path)
        goal_s_expr = (t.get("goal_s_expr") or "").strip()
        match_predicate = _coerce_match_predicate(t.get("match_predicate"))

        if goal_s_expr:
            inner = f"(And {goal_s_expr})"
        elif match_predicate:
            atom = find_goal_atom(parsed["goal_state"], match_predicate)
            if atom is None:
                raise ValueError(
                    f"No goal atom matching predicate '{match_predicate}' in {source_task}"
                )
            inner = f"(And {atom_list_to_s_expr(atom)})"
        else:
            raise ValueError(f"target {tid}: set goal_s_expr or match_predicate")

        raw = open(src_path, "r", encoding="utf-8").read()
        raw = strip_subtask_rewards_block(raw)
        raw = replace_goal_block(raw, inner)
        out_name = f"{_sanitize_filename(tid)}.bddl"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as wf:
            wf.write(raw)
        print(f"[generate] wrote {out_path}")
        written += 1
    print(f"[generate] done, {written} file(s) -> {out_dir}")


def _list_run_files(bddl_dir: str, pattern: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(bddl_dir, pattern)))
    return [p for p in paths if p.endswith(".bddl")]


def _maybe_filter_by_config(
    paths: List[str], config_path: Optional[str]
) -> List[str]:
    if not config_path:
        return paths
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    allowed = {
        _sanitize_filename(t.get("id", "")) + ".bddl"
        for t in (cfg.get("targets") or [])
        if t.get("enabled", True) and t.get("id")
    }
    filtered = [p for p in paths if os.path.basename(p) in allowed]
    return filtered if filtered else paths


def cmd_run(args: argparse.Namespace) -> None:
    from libero.libero.envs.env_wrapper import ControlEnv

    bddl_dir = os.path.abspath(args.bddl_dir or default_benchmark_bddl_dir())
    paths = _list_run_files(bddl_dir, args.glob)
    if args.filter_config:
        paths = _maybe_filter_by_config(paths, args.filter_config)
    if not paths:
        print(f"[run] no .bddl files under {bddl_dir!r} (glob={args.glob!r})")
        return

    rows = []
    for bddl_path in paths:
        print(f"[run] {bddl_path}")
        env = None
        try:
            env = ControlEnv(
                bddl_file_name=bddl_path,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                camera_names=["agentview"],
                camera_heights=64,
                camera_widths=64,
            )
            if args.seed is not None:
                env.seed(args.seed)
            env.reset()
            n_goal_pre = len(env.env.parsed_problem.get("goal_state") or [])
            if n_goal_pre != 1:
                print(
                    f"[run] WARNING: {os.path.basename(bddl_path)} has "
                    f"{n_goal_pre} goal conjunct(s); expected 1 for isolated "
                    f"predicate checks. Tier-A step time scales with all conjuncts."
                )
            action_dim = env.env.action_dim
            action = np.zeros(action_dim, dtype=np.float64)

            for _ in range(args.warmup):
                env.step(action)

            tier_a_times: List[float] = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                for _ in range(args.steps):
                    env.step(action)
                tier_a_times.append((time.perf_counter() - t0) / args.steps)

            mean_a_ms = float(np.mean(tier_a_times)) * 1000.0
            std_a_ms = float(np.std(tier_a_times)) * 1000.0

            n_goal = len(env.env.parsed_problem.get("goal_state") or [])
            pred_label = "|".join(
                str(x[0]).lower() for x in (env.env.parsed_problem.get("goal_state") or [])
            )

            mean_b_ms = std_b_ms = None
            if args.check_loops > 0:
                tier_b_times = []
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    for _ in range(args.check_loops):
                        env.env._check_success()
                    tier_b_times.append((time.perf_counter() - t0) / args.check_loops)
                mean_b_ms = float(np.mean(tier_b_times)) * 1000.0
                std_b_ms = float(np.std(tier_b_times)) * 1000.0

            rows.append(
                {
                    "bddl": os.path.basename(bddl_path),
                    "predicates": pred_label,
                    "n_goal_atoms": n_goal,
                    "tier_a_mean_ms_per_step": f"{mean_a_ms:.4f}",
                    "tier_a_std_ms": f"{std_a_ms:.4f}",
                    "tier_b_mean_ms_per_check": ""
                    if mean_b_ms is None
                    else f"{mean_b_ms:.4f}",
                    "tier_b_std_ms": "" if std_b_ms is None else f"{std_b_ms:.4f}",
                    "steps": args.steps,
                    "check_loops": args.check_loops,
                    "note": "tier_A=full_step; tier_B=_check_success only; subtask_reward=False => ~2*n_goal predicate passes per step",
                }
            )
        except Exception as e:
            print(f"[run] ERROR {bddl_path}: {e}")
            rows.append(
                {
                    "bddl": os.path.basename(bddl_path),
                    "predicates": "",
                    "n_goal_atoms": "",
                    "tier_a_mean_ms_per_step": "ERR",
                    "tier_a_std_ms": str(e),
                    "tier_b_mean_ms_per_check": "",
                    "tier_b_std_ms": "",
                    "steps": args.steps,
                    "check_loops": args.check_loops,
                    "note": "",
                }
            )
        finally:
            if env is not None:
                env.close()

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[run] wrote {args.csv}")

    print("\n[run] summary (Tier A = full env.step per step; Tier B = _check_success per call)")
    for r in rows:
        print(_format_result_row(r))


def main() -> None:
    p = argparse.ArgumentParser(description="LIBERO predicate benchmark utilities")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Write single-predicate-goal BDDLs to predicate_benchmark_temp/")
    g.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "predicate_benchmark_targets.yaml"),
        help="YAML listing targets to materialize",
    )
    g.add_argument(
        "--bddl-dir",
        default=None,
        help="Output directory (default: <bddl_files>/predicate_benchmark_temp)",
    )
    g.add_argument("--clean", action="store_true", help="Remove output directory before writing")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="Time env.step and optional _check_success for each .bddl in dir")
    r.add_argument(
        "--bddl-dir",
        default=None,
        help="Directory containing benchmark .bddl (default: <bddl_files>/predicate_benchmark_temp)",
    )
    r.add_argument("--glob", default="*.bddl", help="Glob under bddl-dir")
    r.add_argument(
        "--filter-config",
        default=None,
        metavar="YAML",
        help="If set, only run .bddl whose basename matches enabled target ids in this YAML",
    )
    r.add_argument("--steps", type=int, default=100, help="Timed env.step count per repeat")
    r.add_argument("--warmup", type=int, default=10, help="Untimed warmup steps before timing")
    r.add_argument("--repeats", type=int, default=3, help="Repeats for mean/std")
    r.add_argument(
        "--check-loops",
        type=int,
        default=500,
        help="Tier B: _check_success calls per repeat (0 to skip)",
    )
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--csv", default=None, help="Optional CSV output path")
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
