"""
Run compute_metrics_parallel_siddhu.py for all (dataset, method, type) combinations
that don't already have results.csv.

GPU-parallel mode:
  --gpus 0,1,2,3  -> spawns pinned worker processes (CUDA_VISIBLE_DEVICES) and
                     schedules tasks across them.
  --workers-per-gpu K -> spawns K workers per GPU (total workers = len(gpus)*K)

CPU-parallel mode (default when --gpus is not set):
  --workers N -> ProcessPoolExecutor like before.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List, Optional, Tuple

# Must match compute_metrics_parallel_siddhu.py
EVAL_BASE = "/home/tedlasai/hdrvideo/evaluations"
DATASETS = ("stuttgart", "ubc")
METHODS = ("lediff", "ours")
RESULTS_FILE = "results.csv"
SCRIPT_NAME = "compute_metrics_parallel_siddhu.py"

Task = Tuple[str, str, str]  # (dataset, method, type)


def _metrics_dir() -> Path:
    return Path(__file__).resolve().parent


def _script_path() -> Path:
    return _metrics_dir() / SCRIPT_NAME


def discover_tasks() -> List[Task]:
    """Scan EVAL_BASE and return (dataset, method, type) where results.csv is missing."""
    tasks: List[Task] = []
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


def run_one_task(task: Task, gpu_id: Optional[str] = None) -> Tuple[Task, bool]:
    """Run compute_metrics_parallel_siddhu.py for one (dataset, method, type). Returns (task, success)."""
    dataset, method, type_name = task

    pred_dir = Path(EVAL_BASE) / f"{method}_{dataset}" / type_name
    if (pred_dir / RESULTS_FILE).is_file():
        return (task, True)  # another process or server wrote it

    script = _script_path()
    if not script.is_file():
        print(f"Missing script: {script}", file=sys.stderr)
        return (task, False)

    cmd = [sys.executable, str(script), dataset, method, type_name]

    env = os.environ.copy()
    if gpu_id is not None:
        # Pin the subprocess to a single GPU.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_metrics_dir()),
            env=env,
            capture_output=False,
            timeout=None,
        )
        return (task, result.returncode == 0)
    except Exception as e:
        print(f"Error running {cmd} (gpu={gpu_id}): {e}", file=sys.stderr)
        return (task, False)


def _gpu_worker(gpu_id: str, task_q: Queue, result_q: Queue) -> None:
    """
    One worker pinned to one GPU. It pulls tasks from task_q and reports to result_q.
    """
    # Ensure any libraries imported in this process see only this GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        task = task_q.get()
        if task is None:
            break

        t, ok = run_one_task(task, gpu_id=gpu_id)
        result_q.put((t, ok, gpu_id))


def _parse_gpus(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    gpus = [x.strip() for x in s.split(",") if x.strip() != ""]
    return gpus if gpus else None


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
        help="Only run this method (e.g. lediff on one server, ours on another).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        default=None,
        help="Only run this dataset.",
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

    # CPU-parallel mode
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU mode: parallel subprocess workers (default 1). Ignored when --gpus is set.",
    )

    # GPU-parallel mode
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        metavar="0,1,2,...",
        help="GPU mode: comma-separated GPU ids. Spawns pinned worker processes.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="GPU mode: number of concurrent workers per GPU (default 1).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print tasks that would be run, then exit.",
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
        print(
            f"\nTotal: {len(tasks)} task(s) missing {RESULTS_FILE}. "
            f"Use --types type1,type2,... to assign."
        )
        return 0

    print(f"Discovered {len(all_tasks)} tasks missing {RESULTS_FILE}; running {len(tasks)} after filters.")
    if not tasks:
        print("Nothing to do.")
        return 0

    for t in tasks:
        print(f"  {t[0]} / {t[1]} / {t[2]}")
    if args.dry_run:
        return 0

    gpus = _parse_gpus(args.gpus)
    failed: List[Task] = []

    # -------------------------
    # GPU-parallel mode
    # -------------------------
    if gpus is not None:
        task_q: Queue = Queue()
        result_q: Queue = Queue()

        workers_per_gpu = max(1, int(args.workers_per_gpu))
        total_workers = len(gpus) * workers_per_gpu

        # Start K workers per GPU
        workers: List[Process] = []
        for gpu_id in gpus:
            for _ in range(workers_per_gpu):
                p = Process(target=_gpu_worker, args=(gpu_id, task_q, result_q), daemon=True)
                p.start()
                workers.append(p)

        # Enqueue tasks
        for t in tasks:
            task_q.put(t)

        # Send stop signals (one per worker process)
        for _ in range(total_workers):
            task_q.put(None)

        # Collect results
        remaining = len(tasks)
        while remaining > 0:
            t, ok, gpu_id = result_q.get()
            dataset, method, type_name = t
            if ok:
                print(f"Done (gpu {gpu_id}): {dataset} / {method} / {type_name}")
            else:
                failed.append(t)
                print(f"Failed (gpu {gpu_id}): {dataset} / {method} / {type_name}", file=sys.stderr)
            remaining -= 1

        # Join workers
        for p in workers:
            p.join()

        if failed:
            print("Failed:", failed, file=sys.stderr)
            return 1
        return 0

    # -------------------------
    # CPU-parallel mode (original behavior)
    # -------------------------
    workers = max(1, min(args.workers, len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one_task, t): t for t in tasks}
        for fut in as_completed(futures):
            (dataset, method, type_name), ok = fut.result()
            if ok:
                print(f"Done: {dataset} / {method} / {type_name}")
            else:
                failed.append((dataset, method, type_name))

    if failed:
        print("Failed:", failed, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())