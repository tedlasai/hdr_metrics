"""
Run compute_metrics_parallel_siddhu.py for all (dataset, method, type) combinations
that don't already have results.csv. Uses ProcessPoolExecutor for parallel runs.

Designed for multi-server: use --method, --dataset, and/or --types to split work
(e.g. server 1: --method lediff; server 2: --method ours --types under,over).
Use --list-types to see all type subfolders (after other filters) then assign with --types.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Must match compute_metrics_parallel_siddhu.py
EVAL_BASE = "/home/tedlasai/hdrvideo/evaluations"
DATASETS = ("stuttgart", "ubc")
METHODS = ("lediff", "ours")
RESULTS_FILE = "results.csv"
SCRIPT_NAME = "compute_metrics_parallel_siddhu.py"


def _metrics_dir():
    return Path(__file__).resolve().parent


def _script_path():
    return _metrics_dir() / SCRIPT_NAME


def discover_tasks():
    """Scan EVAL_BASE and return (dataset, method, type) where results.csv is missing."""
    tasks = []
    base = Path(EVAL_BASE)
    if not base.is_dir():
        return tasks

    for method in METHODS:
        for dataset in DATASETS:
            gt_dir = base / dataset / "hdr"
            if not gt_dir.is_dir():
                continue
            method_dataset = f"{method}_{dataset}"
            pred_parent = base / method_dataset
            if not pred_parent.is_dir():
                continue
            for type_dir in pred_parent.iterdir():
                if not type_dir.is_dir():
                    continue
                results_csv = type_dir / RESULTS_FILE
                if results_csv.is_file():
                    continue
                tasks.append((dataset, method, type_dir.name))
    return sorted(tasks)


def run_one_task(args_tuple):
    """Run compute_metrics_parallel_siddhu.py for one (dataset, method, type). Returns (args_tuple, success)."""
    dataset, method, type_name = args_tuple
    pred_dir = Path(EVAL_BASE) / f"{method}_{dataset}" / type_name
    if (pred_dir / RESULTS_FILE).is_file():
        return (args_tuple, True)  # another process or server wrote it
    script = _script_path()
    if not script.is_file():
        return (args_tuple, False)
    cmd = [sys.executable, str(script), dataset, method, type_name]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_metrics_dir()),
            capture_output=False,
            timeout=None,
        )
        return (args_tuple, result.returncode == 0)
    except Exception as e:
        print(f"Error running {cmd}: {e}", file=sys.stderr)
        return (args_tuple, False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run metrics for all (dataset, method, type) missing results.csv. "
        "Use --method/--dataset/--types to limit work across servers."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=METHODS,
        default=None,
        help="Only run this method (e.g. lediff on one server, ours on another)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        default=None,
        help="Only run this dataset",
    )
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        metavar="T1,T2,...",
        help="Only run these type subfolders (comma-separated). Use --list-types to see available types.",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List all (method, dataset, type) missing results.csv and exit (respects --method/--dataset).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel subprocess workers (default 1; use 1 per server when splitting by method)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print tasks that would be run, then exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_tasks = discover_tasks()
    tasks = all_tasks
    if args.method is not None:
        tasks = [t for t in tasks if t[1] == args.method]
    if args.dataset is not None:
        tasks = [t for t in tasks if t[0] == args.dataset]
    if args.types is not None:
        allowed = {s.strip() for s in args.types.split(",") if s.strip()}
        tasks = [t for t in tasks if t[2] in allowed]
        if not allowed:
            print("No valid types in --types", file=sys.stderr)
            return 1

    if args.list_types:
        seen = set()
        for t in tasks:
            key = (t[1], t[0], t[2])
            if key not in seen:
                seen.add(key)
                print(f"  {t[0]} / {t[1]} / {t[2]}")
        print(f"\nTotal: {len(tasks)} task(s) missing {RESULTS_FILE}. Use --types type1,type2,... to assign.")
        return 0

    print(f"Discovered {len(all_tasks)} tasks missing {RESULTS_FILE}; running {len(tasks)} after filters.")
    if not tasks:
        print("Nothing to do.")
        return 0

    for t in tasks:
        print(f"  {t[0]} / {t[1]} / {t[2]}")
    if args.dry_run:
        return 0

    workers = max(1, min(args.workers, len(tasks)))
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one_task, t): t for t in tasks}
        for fut in as_completed(futures):
            (dataset, method, type_name), ok = fut.result()
            if not ok:
                failed.append((dataset, method, type_name))
            else:
                print(f"Done: {dataset} / {method} / {type_name}")

    if failed:
        print("Failed:", failed, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
