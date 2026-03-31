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
    compute_fid, fid_update, compute_fvd, fvd_update, cvvdp, initialize_cvvdp,fit_and_apply_crf
)

# ----------------------------
# Config
# ----------------------------
EVAL_BASE = "/projects/gencamedit/hdreval/evaluations"
EVAL_OUTPUT_DIR = "/projects/gencamedit/hdreval/evaluations_output"
DATASETS = ("stuttgart", "ubc")
METHODS = ("eilertsen", "lediff", "ours", "hdrtv", "santos", "oursfeb20", "eilertsenfeb")


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
        default=17,
        metavar="N",
        help="Number of frames per video (default 16). Output written to results_{method}_{dataset}_{type}_N_dsK.csv.",
    )
    parser.add_argument(
        "--ds",
        type=int,
        default=1,
        metavar="K",
        help="Downsample: use every K-th frame from each video dir (default 1).",
    )
    parser.add_argument(
        "--videos",
        type=str,
        default=None,
        metavar="V1,V2,...",
        help="Optional comma-separated video folder names to run (default: all videos).",
    )
    parser.add_argument(
        "--videos-exclude",
        type=str,
        default=None,
        metavar="V1,V2,...",
        help="Optional comma-separated video folder names to exclude.",
    )
    return parser.parse_args()


args = parse_args()
gt_dir = os.path.join(EVAL_BASE, args.dataset, "hdr")
pred_dir = os.path.join(EVAL_BASE, f"{args.method}_{args.dataset}", args.type)

print(f"GT Directory: {gt_dir}")
print(f"Pred Directory: {pred_dir}")

NUM_FILES = args.num_files  # max frames per video
DS = max(1, args.ds)  # use every DS-th frame from each video dir

video_paths = sorted([d for d in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, d))])
requested_videos = None
if args.videos:
    requested_videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    requested_set = set(requested_videos)
    video_paths = [v for v in video_paths if v in requested_set]
    if not video_paths:
        raise ValueError(
            f"No matching videos found for --videos={args.videos} under {pred_dir}"
        )
    print(f"Running limited video set ({len(video_paths)}): {video_paths}")
if args.videos_exclude:
    excluded_videos = {v.strip() for v in args.videos_exclude.split(",") if v.strip()}
    video_paths = [v for v in video_paths if v not in excluded_videos]
    if not video_paths:
        raise ValueError(
            f"No videos left after --videos-exclude={args.videos_exclude} under {pred_dir}"
        )
    print(f"Excluding videos: {sorted(excluded_videos)}")

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
reinhard_fvd_first_metric = initialize_fvd()
pu_fvd_first_metric = initialize_fvd()

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


def process_one_frame(idx, pred_im_path, gt_im_path, ab_first, coeffs_first):
    """
    Runs all expensive per-frame computation.
    Returns everything needed by the main thread to:
      - store preds/gts for FVD/CVVDP
      - append scalar metrics
    """
    cv2_hdr_pred_raw = read_image(pred_im_path)
    # cv2_hdr_pred = "frame_0000.exr" #read_image(pred_im_path)
    cv2_hdr_pred = cv2_hdr_pred_raw
    cv2_hdr_gt = read_image(gt_im_path)
    cv2_hdr_gt = pre_hdr_p3(cv2_hdr_gt)
    cv2_hdr_pred, cv2_hdr_gt, _ = align_hdr_pred_to_gt(cv2_hdr_pred, cv2_hdr_gt, percentile_low=20, percentile_high=80)
    cv2_hdr_pred,_ = fit_and_apply_crf(cv2_hdr_pred, cv2_hdr_gt, p_low=0.1, p_high=99.9)

    hdrvdp3_val = hdr_vdp3(cv2_hdr_pred, cv2_hdr_gt)

    pu_pred, pu_gt = pu(cv2_hdr_pred), pu(cv2_hdr_gt)

    cv2_hdr_pred_f_aligned, _, _ = align_hdr_pred_to_gt(
        cv2_hdr_pred_raw,
        cv2_hdr_gt,
        ab=ab_first,
    )

    # 2) fixed CRF apply
    cv2_hdr_pred_f, _ = fit_and_apply_crf(
        cv2_hdr_pred_f_aligned,
        cv2_hdr_gt,
        coeffs=coeffs_first,
    )

    # 3) Reinhard + PU for FVD
    reinhard_pred_f = reinhard_tonemap(cv2_hdr_pred_f)

    denom = np.max(pu_gt) if np.max(pu_gt) > 0 else 1.0
    pu_pred_f = pu(cv2_hdr_pred_f)
    pu_pred_norm_f = pu_pred_f / denom  # same denom as your normal path (pu_gt)
    pu_pred_norm = pu_pred / denom
    pu_gt_norm = pu_gt / denom
    
    reinhard_pred = reinhard_tonemap(cv2_hdr_pred)
    reinhard_gt   = reinhard_tonemap(cv2_hdr_gt)

    pu_pred_01 = np.clip(pu_pred_norm, 0.0, 1.0).astype(np.float32)
    pu_gt_01 = np.clip(pu_gt_norm, 0.0, 1.0).astype(np.float32)

    # Scalar metrics
    pu_psnr = psnr(pu_pred, pu_gt)
    pu_vsi  = vsi(pu_pred, pu_gt)
    pu_piqe = piqe(pu_pred)
    with lpips_lock:
        pu_lpips = lpips(pu_pred_01, pu_gt_01)
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
        "reinhard_pred_firstfit": reinhard_pred_f,
        "pu_pred_norm_firstfit": pu_pred_norm_f,
    }


for video_path in tqdm(video_paths, desc="Videos", unit="video"):
    pred_video_dir = os.path.join(pred_dir, video_path)
    gt_video_dir   = os.path.join(gt_dir, video_path)

    all_frames = sorted(os.listdir(pred_video_dir))
    im_paths = all_frames[::DS][:NUM_FILES]
    assert len(im_paths) >= 1, (
        f"Expected at least 1 frame after downsampling (ds={DS}), found {len(im_paths)} in {pred_video_dir}"
    )

    ALIGN_P_LO, ALIGN_P_HI = 20, 80 # 160
    CRF_P_LO, CRF_P_HI = 0.1, 99.9 #161

    im0 = im_paths[0]
    pred0 = read_image(os.path.join(pred_video_dir, im0))
    gt0 = pre_hdr_p3(read_image(os.path.join(gt_video_dir, im0)))

    pred0_aligned, gt0_aligned, ab_first = align_hdr_pred_to_gt(
        pred0, gt0,
        percentile_low=ALIGN_P_LO,
        percentile_high=ALIGN_P_HI,
    )

    _, coeffs_first = fit_and_apply_crf(
        pred0_aligned, gt0_aligned,
        p_low=CRF_P_LO,
        p_high=CRF_P_HI,
    )

    # Per-video scalar metric accumulators (written to per-video CSV)
    v_psnr_scores = []
    v_vsi_scores = []
    v_piqe_scores = []
    v_lpips_scores = []
    v_hdrvdp3_scores = []

    # Pre-allocate lists so you keep order for video metrics
    reinhard_preds = [None] * len(im_paths)
    reinhard_gts   = [None] * len(im_paths)
    reinhard_preds_firstfit = [None] * len(im_paths)
    reinhard_gts_firstfit = [None] * len(im_paths)
    pu_preds       = [None] * len(im_paths)
    pu_gts         = [None] * len(im_paths)
    pu_preds_firstfit = [None] * len(im_paths)
    pu_gts_firstfit = [None] * len(im_paths)
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
            fut = ex.submit(process_one_frame, idx, pred_im_path, gt_im_path, ab_first, coeffs_first)
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
                reinhard_preds_firstfit[idx] = out["reinhard_pred_firstfit"]
                pu_preds_firstfit[idx] = out["pu_pred_norm_firstfit"]

                # Append scalar metrics
                psnr_scores.append(out["pu_psnr"])
                vsi_scores.append(out["pu_vsi"])
                piqe_scores.append(out["pu_piqe"])
                lpips_scores.append(out["pu_lpips"])
                hdrvdp3_scores.append(out["hdrvdp3"])

                v_psnr_scores.append(out["pu_psnr"])
                v_vsi_scores.append(out["pu_vsi"])
                v_piqe_scores.append(out["pu_piqe"])
                v_lpips_scores.append(out["pu_lpips"])
                v_hdrvdp3_scores.append(out["hdrvdp3"])

                # Drop references ASAP
                del out
                pbar.update(1)

                # submit next frame (if any)
                try:
                    idx2, im_name2 = next(it)
                    pred_im_path2 = os.path.join(pred_video_dir, im_name2)
                    gt_im_path2   = os.path.join(gt_video_dir, im_name2)
                    fut2 = ex.submit(process_one_frame, idx2, pred_im_path2, gt_im_path2, ab_first, coeffs_first)
                    futures[fut2] = idx2
                except StopIteration:
                    pass

                # break so you re-enter while loop with updated futures
                break

        pbar.close()

    # ----------------------------
    # Update per-video metrics (main thread): CVVDP, then full-sequence FID/FVD
    # ----------------------------
    print("Computing CVVDP for video:", video_path)
    cvvdp_score = cvvdp(hdr_preds, hdr_gts, cvvdp_metric)
    cvvdp_scores.append(cvvdp_score)
    print("CVVDP Score:", cvvdp_score)

    # Write per-video CSV (matches historical per-video output format)
    out_csv_video = (
        Path(EVAL_OUTPUT_DIR)
        / f"results_{args.method}_{args.dataset}_{args.type}_{video_path}_{NUM_FILES}_ds{DS}.csv"
    )
    Path(EVAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    video_fieldnames = ["PU-PSNR", "PU-VSI", "PIQE", "LPIPS", "HDR-VDP3", "CVVDP"]
    video_row = {
        "PU-PSNR": float(np.mean(np.array(v_psnr_scores))) if len(v_psnr_scores) else float("nan"),
        "PU-VSI": float(np.mean(np.array(v_vsi_scores))) if len(v_vsi_scores) else float("nan"),
        "PIQE": float(np.mean(np.array(v_piqe_scores))) if len(v_piqe_scores) else float("nan"),
        "LPIPS": float(np.mean(np.array(v_lpips_scores))) if len(v_lpips_scores) else float("nan"),
        "HDR-VDP3": float(np.mean(np.array(v_hdrvdp3_scores))) if len(v_hdrvdp3_scores) else float("nan"),
        "CVVDP": float(cvvdp_score),
    }
    with open(out_csv_video, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=video_fieldnames)
        writer.writeheader()
        writer.writerow(video_row)

    print(f"Wrote per-video: {out_csv_video}")

    print("Updating FID for video:", video_path)
    fid_update(reinhard_preds, reinhard_gts, reinhard_fid_metric)
    fid_update(pu_preds, pu_gts, pu_fid_metric)

    print("Updating FVD for video:", video_path)
    fvd_update(reinhard_preds, reinhard_gts, reinhard_fvd_metric)
    fvd_update(pu_preds, pu_gts, pu_fvd_metric)
    fvd_update(reinhard_preds_firstfit, reinhard_gts, reinhard_fvd_first_metric)
    fvd_update(pu_preds_firstfit, pu_gts, pu_fvd_first_metric)

    del reinhard_preds, reinhard_gts, pu_preds, pu_gts, hdr_preds, hdr_gts, im_paths
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
# --- compute aggregates once (single compute per metric; state is consumed) ---
R_FID   = float(compute_fid(reinhard_fid_metric))
R_FVD   = float(compute_fvd(reinhard_fvd_metric))
PU_FID  = float(compute_fid(pu_fid_metric))
PU_FVD  = float(compute_fvd(pu_fvd_metric))
R_FVD_FIRST = float(compute_fvd(reinhard_fvd_first_metric))
PU_FVD_FIRST = float(compute_fvd(pu_fvd_first_metric))


print("R-FID Score:", R_FID)
print("R-FVD Score:", R_FVD)
print("PU-FID Score:", PU_FID)
print("PU-FVD Score:", PU_FVD)
print("R-FVD-FIRST Score:", R_FVD_FIRST)
print("PU-FVD-FIRST Score:", PU_FVD_FIRST)
print("Average PU-PSNR:", float(np.mean(np.array(psnr_scores))))
print("Average PU-VSI:",  float(np.mean(np.array(vsi_scores))))
print("Average PIQE:",    float(np.mean(np.array(piqe_scores))))
print("Average LPIPS:",   float(np.mean(np.array(lpips_scores))))
print("Average HDR-VDP3:", float(np.mean(np.array(hdrvdp3_scores))))
print("Average CVVDP:", float(np.mean(np.array(cvvdp_scores))))

PU_PSNR = float(np.mean(np.array(psnr_scores))) if len(psnr_scores) else float("nan")
PU_VSI  = float(np.mean(np.array(vsi_scores)))  if len(vsi_scores)  else float("nan")
PIQE    = float(np.mean(np.array(piqe_scores))) if len(piqe_scores) else float("nan")
LPIPS   = float(np.mean(np.array(lpips_scores))) if len(lpips_scores) else float("nan")
HDR_VDP3 = float(np.mean(np.array(hdrvdp3_scores))) if len(hdrvdp3_scores) else float("nan")
CVVDP   = float(np.mean(np.array(cvvdp_scores))) if len(cvvdp_scores) else float("nan")


# --- write CSV ---
out_csv = Path(EVAL_OUTPUT_DIR) / f"results_{args.method}_{args.dataset}_{args.type}_{NUM_FILES}_ds{DS}.csv"
Path(EVAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
fieldnames = [
    "CVVDP", "HDR-VDP3", "PU-PSNR", "PU-VSI", "PIQE", "LPIPS",
    "R-FID", "R-FVD", "PU-FID", "PU-FVD", "R-FVD-FIRST", "PU-FVD-FIRST",
]
row = {
    "CVVDP": CVVDP, "HDR-VDP3": HDR_VDP3, "PU-PSNR": PU_PSNR, "PU-VSI": PU_VSI, "PIQE": PIQE, "LPIPS": LPIPS,
    "R-FID": R_FID, "R-FVD": R_FVD, "PU-FID": PU_FID, "PU-FVD": PU_FVD, "R-FVD-FIRST": R_FVD_FIRST, "PU-FVD-FIRST": PU_FVD_FIRST
}

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(row)

print(f"Wrote: {out_csv}")
