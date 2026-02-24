# CONDA ENVIRONMENT P312
import argparse
import os
import csv
import gc
import threading
from pathlib import Path
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from tqdm import tqdm  # progress bars

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# Reduce CUDA fragmentation (helps with OOM when many models + threads use GPU)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from metrics_helpers import (
    read_image, pre_hdr_p3, align_hdr_pred_to_gt,
    psnr, vsi, piqe, lpips, hdr_vdp3, pu, reinhard_tonemap,
    initialize_fid, initialize_fvd,
    compute_fid, fid_update, compute_fvd, fvd_update, cvvdp, initialize_cvvdp
)

# ----------------------------
# Config
# ----------------------------
EVAL_BASE = "/home/tedlasai/hdrvideo/evaluations"
DATASETS = ("stuttgart", "ubc")
METHODS = ("eilertsen", "lediff", "ours")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute HDR video metrics")
    parser.add_argument(
        "dataset",
        type=str,
        choices=DATASETS,
        help="Dataset: stuttgart or ubc",
    )
    parser.add_argument(
        "method",
        type=str,
        choices=METHODS,
        help="Method name, e.g. lediff or ours",
    )
    parser.add_argument(
        "type",
        type=str,
        help="Subfolder under method (e.g. under)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=16,
        metavar="N",
        help="Number of frames per video (default 16). Output written to resultsN.csv.",
    )
    return parser.parse_args()


args = parse_args()
gt_dir = os.path.join(EVAL_BASE, args.dataset, "hdr")
pred_dir = os.path.join(EVAL_BASE, f"{args.method}_{args.dataset}", args.type)

print(f"GT Directory: {gt_dir}")
print(f"Pred Directory: {pred_dir}")

NUM_FILES = args.num_files  # e.g. 16 for testing, 200 for full run; output -> results{NUM_FILES}.csv

video_paths = sorted([d for d in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, d))])


def _int_env(name: str, default: int, min_val: int = 1, max_val: int = 32) -> int:
    """Read int from env with clamp; used so parent can force small workers via env."""
    try:
        v = int(os.environ.get(name, default))
        return max(min_val, min(max_val, v))
    except (TypeError, ValueError):
        return default


def get_cuda_free_mb() -> Optional[float]:
    """Return free GPU memory in MB, or None if CUDA not used."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        if hasattr(torch.cuda, "mem_get_info"):
            free, _ = torch.cuda.mem_get_info()
            return free / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        return (total - reserved) / (1024 ** 2)
    except Exception:
        return None


def _cuda_clear() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


# Concurrency: parent (metric_gathering_sai) sets METRICS_MAX_WORKERS / METRICS_MAX_IN_FLIGHT
# so --workers-per-gpu 1 or 2 actually limits per-process concurrency.
max_workers = _int_env("METRICS_MAX_WORKERS", 1)
MAX_IN_FLIGHT = _int_env("METRICS_MAX_IN_FLIGHT", 2)
SAFETY_MB = _int_env("METRICS_SAFETY_MB", 1500, min_val=256, max_val=16000)

# Log GPU and limits (so you can confirm 1 GPU and small worker count)
_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
print(f"CUDA_VISIBLE_DEVICES={_visible} | METRICS_MAX_WORKERS={max_workers} MAX_IN_FLIGHT={MAX_IN_FLIGHT}")
free_mb = get_cuda_free_mb()
if free_mb is not None:
    print(f"GPU free memory at start: {free_mb:.0f} MB; backpressure threshold: {SAFETY_MB} MB")

# ----------------------------
# Initialize metrics (shared state)
# ----------------------------
cvvdp_metric = initialize_cvvdp()
reinhard_fvd_metric = initialize_fvd()
reinhard_fid_metric = initialize_fid()
pu_fvd_metric = initialize_fvd()
pu_fid_metric = initialize_fid()

# If fid_update is not thread-safe, you update it in the main thread (you already do).
lpips_lock = threading.Lock()

# ----------------------------
# Accumulators
# ----------------------------
psnr_scores = []
vsi_scores = []
piqe_scores = []
lpips_scores = []
hdrvdp3_scores = []
cvvdp_scores = []


def process_one_frame(idx, pred_im_path, gt_im_path):
    """
    Runs all expensive per-frame computation.
    Returns everything needed by the main thread to:
      - store preds/gts for FVD/CVVDP
      - append scalar metrics
    """
    cv2_hdr_pred = read_image(pred_im_path)
    cv2_hdr_gt = read_image(gt_im_path)
    cv2_hdr_gt = pre_hdr_p3(cv2_hdr_gt)
    cv2_hdr_pred, cv2_hdr_gt, _ = align_hdr_pred_to_gt(cv2_hdr_pred, cv2_hdr_gt)

    hdrvdp3_val = hdr_vdp3(cv2_hdr_pred, cv2_hdr_gt)

    pu_pred, pu_gt = pu(cv2_hdr_pred), pu(cv2_hdr_gt)

    # Normalize PU frames by max(pu_gt)
    denom = np.max(pu_gt) if np.max(pu_gt) > 0 else 1.0
    pu_pred_norm = pu_pred / denom
    pu_gt_norm = pu_gt / denom

    reinhard_pred = reinhard_tonemap(cv2_hdr_pred)
    reinhard_gt   = reinhard_tonemap(cv2_hdr_gt)

    # Scalar metrics
    pu_psnr = psnr(pu_pred, pu_gt)
    pu_vsi  = vsi(pu_pred, pu_gt)
    pu_piqe = piqe(pu_pred)
    with lpips_lock:
        pu_lpips = lpips(reinhard_pred, reinhard_gt)

    return {
        "idx": idx,
        "cv2_hdr_pred": cv2_hdr_pred,
        "cv2_hdr_gt": cv2_hdr_gt,
        "reinhard_pred": reinhard_pred,
        "reinhard_gt": reinhard_gt,
        "pu_pred_norm": pu_pred_norm,
        "pu_gt_norm": pu_gt_norm,
        "pu_psnr": pu_psnr,
        "pu_vsi": pu_vsi,
        "pu_piqe": pu_piqe,
        "pu_lpips": pu_lpips,
        "hdrvdp3": hdrvdp3_val,
    }


for video_path in tqdm(video_paths, desc="Videos", unit="video"):
    pred_video_dir = os.path.join(pred_dir, video_path)
    gt_video_dir   = os.path.join(gt_dir, video_path)

    im_paths = sorted(os.listdir(pred_video_dir))[:NUM_FILES]
    assert len(im_paths) == NUM_FILES, (
        f"Expected {NUM_FILES} frames, found {len(im_paths)} in {pred_video_dir}"
    )

    # Pre-allocate lists so you keep order for video metrics
    reinhard_preds = [None] * len(im_paths)
    reinhard_gts   = [None] * len(im_paths)
    pu_preds       = [None] * len(im_paths)
    pu_gts         = [None] * len(im_paths)
    hdr_preds      = [None] * len(im_paths)
    hdr_gts        = [None] * len(im_paths)

    # ----------------------------
    # Bounded in-flight futures
    # ----------------------------
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        it = enumerate(im_paths)

        futures = {}

        # submit initial window
        for idx, im_name in islice(it, MAX_IN_FLIGHT):
            pred_im_path = os.path.join(pred_video_dir, im_name)
            gt_im_path   = os.path.join(gt_video_dir, im_name)
            fut = ex.submit(process_one_frame, idx, pred_im_path, gt_im_path)
            futures[fut] = idx

        pbar = tqdm(total=len(im_paths), desc=f"Frames [{video_path}]", unit="frame", leave=False)

        while futures:
            # wait for at least one future to complete
            for fut in as_completed(list(futures.keys())):
                _ = futures.pop(fut)  # idx not strictly needed; out contains idx
                out = fut.result()
                idx = out["idx"]

                # Store for video-level metrics (keep order)
                reinhard_preds[idx] = out["reinhard_pred"]
                reinhard_gts[idx]   = out["reinhard_gt"]
                pu_preds[idx]       = out["pu_pred_norm"]
                pu_gts[idx]         = out["pu_gt_norm"]
                hdr_preds[idx]      = out["cv2_hdr_pred"]
                hdr_gts[idx]        = out["cv2_hdr_gt"]

                # Append scalar metrics
                psnr_scores.append(out["pu_psnr"])
                vsi_scores.append(out["pu_vsi"])
                piqe_scores.append(out["pu_piqe"])
                lpips_scores.append(out["pu_lpips"])
                hdrvdp3_scores.append(out["hdrvdp3"])

                # Drop references ASAP
                del out
                pbar.update(1)

                # submit next frame (if any)
                try:
                    idx2, im_name2 = next(it)
                    pred_im_path2 = os.path.join(pred_video_dir, im_name2)
                    gt_im_path2   = os.path.join(gt_video_dir, im_name2)
                    fut2 = ex.submit(process_one_frame, idx2, pred_im_path2, gt_im_path2)
                    futures[fut2] = idx2
                except StopIteration:
                    pass

                # break so you re-enter while loop with updated futures
                break

        pbar.close()

    # ----------------------------
    # Update per-video metrics. Clear GPU after frame phase so video-phase has headroom.
    # Run CVVDP then FID/FVD. FID and FVD are updated in small chunks so we never
    # put the full video on GPU at once (avoids 18+ GiB spike and OOM).
    # ----------------------------
    gc.collect()
    _cuda_clear()

    # Frames per chunk for FID/FVD to cap peak GPU memory (env: METRICS_VIDEO_CHUNK)
    video_chunk = _int_env("METRICS_VIDEO_CHUNK", 4, min_val=1, max_val=64)

    print("Computing CVVDP for video:", video_path)
    cvvdp_score = cvvdp(hdr_preds, hdr_gts, cvvdp_metric)
    cvvdp_scores.append(cvvdp_score)
    print("CVVDP Score:", cvvdp_score)
    del hdr_preds, hdr_gts
    gc.collect()
    _cuda_clear()

    print("Updating FID for video:", video_path)
    for start in range(0, len(reinhard_preds), video_chunk):
        end = start + video_chunk
        fid_update(reinhard_preds[start:end], reinhard_gts[start:end], reinhard_fid_metric)
        _cuda_clear()
    for start in range(0, len(pu_preds), video_chunk):
        end = start + video_chunk
        fid_update(pu_preds[start:end], pu_gts[start:end], pu_fid_metric)
        _cuda_clear()

    print("Updating FVD for video:", video_path)
    for start in range(0, len(reinhard_preds), video_chunk):
        end = start + video_chunk
        fvd_update(reinhard_preds[start:end], reinhard_gts[start:end], reinhard_fvd_metric)
        _cuda_clear()
    for start in range(0, len(pu_preds), video_chunk):
        end = start + video_chunk
        fvd_update(pu_preds[start:end], pu_gts[start:end], pu_fvd_metric)
        _cuda_clear()

    del reinhard_preds, reinhard_gts, pu_preds, pu_gts, im_paths
    gc.collect()
    _cuda_clear()

# ----------------------------
# Final metrics
# ----------------------------
print("R-FID Score:", compute_fid(reinhard_fid_metric))
print("R-FVD Score:", compute_fvd(reinhard_fvd_metric))
print("PU-FID Score:", compute_fid(pu_fid_metric))
print("PU-FVD Score:", compute_fvd(pu_fvd_metric))

print("Average PU-PSNR:", float(np.mean(np.array(psnr_scores))))
print("Average PU-VSI:",  float(np.mean(np.array(vsi_scores))))
print("Average PIQE:",    float(np.mean(np.array(piqe_scores))))
print("Average LPIPS:",   float(np.mean(np.array(lpips_scores))))
print("Average HDR-VDP3:", float(np.mean(np.array(hdrvdp3_scores))))
print("Average CVVDP:", float(np.mean(np.array(cvvdp_scores))))

# --- compute aggregates once ---
R_FID   = float(compute_fid(reinhard_fid_metric))
R_FVD   = float(compute_fvd(reinhard_fvd_metric))
PU_FID  = float(compute_fid(pu_fid_metric))
PU_FVD  = float(compute_fvd(pu_fvd_metric))

PU_PSNR = float(np.mean(np.array(psnr_scores))) if len(psnr_scores) else float("nan")
PU_VSI  = float(np.mean(np.array(vsi_scores)))  if len(vsi_scores)  else float("nan")
PIQE    = float(np.mean(np.array(piqe_scores))) if len(piqe_scores) else float("nan")
LPIPS   = float(np.mean(np.array(lpips_scores))) if len(lpips_scores) else float("nan")
HDR_VDP3 = float(np.mean(np.array(hdrvdp3_scores))) if len(hdrvdp3_scores) else float("nan")
CVVDP   = float(np.mean(np.array(cvvdp_scores))) if len(cvvdp_scores) else float("nan")

# (optional) also save std-devs for sanity
PU_PSNR_s  = float(np.std(np.array(psnr_scores))) if len(psnr_scores) else float("nan")
PU_VSI_s   = float(np.std(np.array(vsi_scores)))  if len(vsi_scores)  else float("nan")
PIQE_s     = float(np.std(np.array(piqe_scores))) if len(piqe_scores) else float("nan")
LPIPS_s    = float(np.std(np.array(lpips_scores))) if len(lpips_scores) else float("nan")
HDR_VDP3_s = float(np.std(np.array(hdrvdp3_scores))) if len(hdrvdp3_scores) else float("nan")
CVVDP_s    = float(np.std(np.array(cvvdp_scores))) if len(cvvdp_scores) else float("nan")

# --- write CSV ---
out_csv = Path(pred_dir) / f"results{NUM_FILES}.csv"
fieldnames = [
    "CVVDP", "HDR-VDP3", "PU-PSNR", "PU-VSI", "PIQE", "LPIPS",
    "R-FID", "R-FVD", "PU-FID", "PU-FVD",
    "CVVDP-STD", "HDR-VDP3-STD", "PU-PSNR-STD", "PU-VSI-STD", "PIQE-STD", "LPIPS-STD",
]
row = {
    "CVVDP": CVVDP, "HDR-VDP3": HDR_VDP3, "PU-PSNR": PU_PSNR, "PU-VSI": PU_VSI, "PIQE": PIQE, "LPIPS": LPIPS,
    "R-FID": R_FID, "R-FVD": R_FVD, "PU-FID": PU_FID, "PU-FVD": PU_FVD,
    "CVVDP-STD": CVVDP_s, "HDR-VDP3-STD": HDR_VDP3_s, "PU-PSNR-STD": PU_PSNR_s,
    "PU-VSI-STD": PU_VSI_s, "PIQE-STD": PIQE_s, "LPIPS-STD": LPIPS_s
}

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(row)

print(f"Wrote: {out_csv}")
