# CONDA ENVIRONMENT P312
import argparse
import os
import csv
import gc
import threading
from pathlib import Path
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm  # progress bars

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from metrics_helpers import (
    read_image, pre_hdr_p3, align_hdr_pred_to_gt,
    psnr, vsi, piqe, lpips, hdr_vdp3, pu, reinhard_tonemap,
    initialize_fid, initialize_fvd,
    compute_fid, fid_update, compute_fvd, fvd_update, cvvdp, initialize_cvvdp
)

# ----------------------------
# Config (CLI for metric_gathering_sai_reinhard.py)
# ----------------------------
EVAL_BASE = "/projects/gencamedit/hdreval/evaluations"
EVAL_OUTPUT_DIR = "/projects/gencamedit/hdreval/evaluations_output"
DATASETS = ("stuttgart", "ubc")
METHODS = ("eilertsen", "lediff", "ours", "hdrtv", "santos", "oursfeb20", "eilertsenfeb")


def _int_env(name: str, default: int, min_val: int = 1, max_val: int = 32) -> int:
    try:
        v = int(os.environ.get(name, default))
        return max(min_val, min(max_val, v))
    except (TypeError, ValueError):
        return default


def parse_args():
    parser = argparse.ArgumentParser(description="Compute HDR video metrics (Reinhard tonemap)")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="Dataset: stuttgart or ubc")
    parser.add_argument("method", type=str, choices=METHODS, help="Method name")
    parser.add_argument("type", type=str, help="Subfolder under method (e.g. under)")
    parser.add_argument("--num-files", type=int, default=17, metavar="N", help="Max frames per video (default 17).")
    parser.add_argument("--ds", type=int, default=1, metavar="K", help="Use every K-th frame (default 1).")
    return parser.parse_args()


args = parse_args()
gt_dir = os.path.join(EVAL_BASE, args.dataset, "hdr")
pred_dir = os.path.join(EVAL_BASE, f"{args.method}_{args.dataset}", args.type)
NUM_FILES = args.num_files
DS = max(1, args.ds)

video_paths = sorted([d for d in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, d))])

max_workers = _int_env("METRICS_MAX_WORKERS", 1)
MAX_IN_FLIGHT = _int_env("METRICS_MAX_IN_FLIGHT", 2)

# ----------------------------
# Initialize metrics (shared state)
# ----------------------------
cvvdp_metric = initialize_cvvdp()
reinhard_fvd_metric = initialize_fvd()
reinhard_fid_metric = initialize_fid()
# pu_fvd_metric = initialize_fvd()
# pu_fid_metric = initialize_fid()

# If fid_update is not thread-safe, you update it in the main thread (you already do).
# lpips_lock = threading.Lock()

# ----------------------------
# Accumulators
# ----------------------------
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

    reinhard_pred = reinhard_tonemap(cv2_hdr_pred)
    reinhard_gt   = reinhard_tonemap(cv2_hdr_gt)

    # reinhard_gt = pre_hdr_p3(reinhard_gt)
    # reinhard_pred, reinhard_gt, _ = align_hdr_pred_to_gt(reinhard_pred, reinhard_gt)


    # # Scalar metrics
    # pu_psnr = psnr(pu_pred, pu_gt)
    # pu_vsi  = vsi(pu_pred, pu_gt)
    # pu_piqe = piqe(pu_pred)
    # with lpips_lock:
    #     pu_lpips = lpips(reinhard_pred, reinhard_gt)

    return {
        "idx": idx,
        "cv2_hdr_pred": cv2_hdr_pred,
        "cv2_hdr_gt": cv2_hdr_gt,
        "reinhard_pred": reinhard_pred,
        "reinhard_gt": reinhard_gt,
        # "pu_pred_norm": pu_pred_norm,
        # "pu_gt_norm": pu_gt_norm,
        # "pu_psnr": pu_psnr,
        # "pu_vsi": pu_vsi,
        # "pu_piqe": pu_piqe,
        # "pu_lpips": pu_lpips,
        # "hdrvdp3": hdrvdp3_val,
    }


for video_path in tqdm(video_paths, desc="Videos", unit="video"):
    pred_video_dir = os.path.join(pred_dir, video_path)
    gt_video_dir   = os.path.join(gt_dir, video_path)

    all_ims = sorted(os.listdir(pred_video_dir))
    im_paths = all_ims[::DS][:NUM_FILES]
    assert len(im_paths) >= 1, (
        f"Need at least 1 frame after DS={DS}, found {len(im_paths)} in {pred_video_dir}"
    )

    # Pre-allocate lists so you keep order for video metrics
    reinhard_preds = [None] * len(im_paths)
    reinhard_gts   = [None] * len(im_paths)
    # pu_preds       = [None] * len(im_paths)
    # pu_gts         = [None] * len(im_paths)
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
                hdr_preds[idx]      = out["cv2_hdr_pred"]
                hdr_gts[idx]        = out["cv2_hdr_gt"]

                # # Append scalar metrics
                # psnr_scores.append(out["pu_psnr"])
                # vsi_scores.append(out["pu_vsi"])
                # piqe_scores.append(out["pu_piqe"])
                # lpips_scores.append(out["pu_lpips"])
                # hdrvdp3_scores.append(out["hdrvdp3"])

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
    # Update per-video metrics (main thread)
    # ----------------------------
    cvvdp_score = cvvdp(hdr_preds, hdr_gts, cvvdp_metric)
    cvvdp_scores.append(cvvdp_score)
    fid_update(reinhard_preds, reinhard_gts, reinhard_fid_metric)
    fvd_update(reinhard_preds, reinhard_gts, reinhard_fvd_metric)
    # fvd_update(pu_preds, pu_gts, pu_fvd_metric)

    # Cleanup
    del reinhard_preds, reinhard_gts, hdr_preds, hdr_gts, im_paths
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

# ----------------------------
# Final metrics
# ----------------------------
# print("PU-FID Score:", compute_fid(pu_fid_metric))
# print("PU-FVD Score:", compute_fvd(pu_fvd_metric))

# print("Average PU-PSNR:", float(np.mean(np.array(psnr_scores))))
# print("Average PU-VSI:",  float(np.mean(np.array(vsi_scores))))
# print("Average PIQE:",    float(np.mean(np.array(piqe_scores))))
# print("Average LPIPS:",   float(np.mean(np.array(lpips_scores))))
# print("Average HDR-VDP3:", float(np.mean(np.array(hdrvdp3_scores))))
# print("Average CVVDP:", float(np.mean(np.array(cvvdp_scores))))

# --- compute aggregates once ---
R_FID   = float(compute_fid(reinhard_fid_metric))
R_FVD   = float(compute_fvd(reinhard_fvd_metric))
# PU_FID  = float(compute_fid(pu_fid_metric))
# PU_FVD  = float(compute_fvd(pu_fvd_metric))

# PU_PSNR = float(np.mean(np.array(psnr_scores))) if len(psnr_scores) else float("nan")
# PU_VSI  = float(np.mean(np.array(vsi_scores)))  if len(vsi_scores)  else float("nan")
# PIQE    = float(np.mean(np.array(piqe_scores))) if len(piqe_scores) else float("nan")
# LPIPS   = float(np.mean(np.array(lpips_scores))) if len(lpips_scores) else float("nan")
# HDR_VDP3 = float(np.mean(np.array(hdrvdp3_scores))) if len(hdrvdp3_scores) else float("nan")
# CVVDP   = float(np.mean(np.array(cvvdp_scores))) if len(cvvdp_scores) else float("nan")

# (optional) also save std-devs for sanity
# PU_PSNR_s  = float(np.std(np.array(psnr_scores))) if len(psnr_scores) else float("nan")
# PU_VSI_s   = float(np.std(np.array(vsi_scores)))  if len(vsi_scores)  else float("nan")
# PIQE_s     = float(np.std(np.array(piqe_scores))) if len(piqe_scores) else float("nan")
# LPIPS_s    = float(np.std(np.array(lpips_scores))) if len(lpips_scores) else float("nan")
# HDR_VDP3_s = float(np.std(np.array(hdrvdp3_scores))) if len(hdrvdp3_scores) else float("nan")
# CVVDP_s    = float(np.std(np.array(cvvdp_scores))) if len(cvvdp_scores) else float("nan")

# --- write CSV: only CVVDP, R-FID, R-FVD (metric_gathering_sai_reinhard looks for *_reinhard.csv) ---
Path(EVAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
out_csv = Path(EVAL_OUTPUT_DIR) / f"results_{args.method}_{args.dataset}_{args.type}_{NUM_FILES}_ds{DS}_reinhard.csv"
fieldnames = ["CVVDP", "R-FID", "R-FVD"]
row = {"R-FID": R_FID, "R-FVD": R_FVD}
if cvvdp_scores:
    row["CVVDP"] = float(np.mean(cvvdp_scores))
else:
    row["CVVDP"] = ""
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(row)
