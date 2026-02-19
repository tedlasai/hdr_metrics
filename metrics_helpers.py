import sys
path_hdr_vdp3 = "./jpeg-ai-qaf-feature-HDR-VDP2.2-main"
sys.path.append(path_hdr_vdp3)  
from VDP3.hdrvdp3 import hdrvdp3
from pu21 import PU21Encoder
from vsi import m_vsi
from NRVQA.piqe import piqe as piqe_func
pu21 = PU21Encoder("banding_glare")
import torch

# PyTorch 2.6+ defaults to weights_only=True; cdfvd and other checkpoints need weights_only=False
_orig_torch_load = torch.load
def _torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from cdfvd import fvd
import pycvvdp

lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)

def read_image(path):
    if path.endswith(".hdr") or path.endswith(".exr"):
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    img = np.clip(img, 0, None)
    return img

def initialize_cvvdp(display_name='ours_standard_hdr_linear', heatmap='threshold'):
    cvvdp = pycvvdp.cvvdp(display_name=display_name, heatmap=heatmap)
    return cvvdp

def initialize_fid(device="cuda"):
    fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(device)
    return fid_metric

def initialize_fvd(device="cuda"):
    fvd_metric = fvd.cdfvd(model='videomae', device=device)
    return fvd_metric

def pu(img):
    return pu21.encode(img).astype(np.float32)

def reinhard_tonemap(img):
    return img / (1.0 + img)

@torch.no_grad()
def lpips(pred, gt, device="cuda"):

    pred = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0).to(device).float()
    gt   = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).to(device).float()

    lpips_metric.to(device)
    return float(lpips_metric(pred, gt).item())

@torch.no_grad()
def cvvdp(preds, gt, metric):
    preds = np.stack(preds, axis=0)
    gts = np.stack(gt, axis=0)
    return metric.predict(preds, gts, dim_order="FHWC", frames_per_second=30)[0].item()

def hdr_vdp3(pred, gt): 
    pred = np.clip(pred, 0, None)
    gt = np.clip(gt, 0, None)
    jod = hdrvdp3('quality', pred, gt, 'rgb-bt.709', 30)
    return jod['Q_JOD']

def psnr(pred, gt):
    mse = np.mean((pred - gt)**2)
    return 10 * np.log10((256**2) / mse)

def vsi(pred, gt):
    return m_vsi(pred, gt)

def piqe(pred):
    piqe_out = piqe_func(pred.astype(np.float32))
    return piqe_out[0]

# def pu_piqe(pred):
#     pu21_pred = pu21.encode(pred)
#     piqe_out = piqe(pu21_pred.astype(np.float32))
#     return piqe_out[0]

@torch.no_grad()
def fid_update(pred_video, gt_video, fid_metric):
    pred_video = np.stack(pred_video, axis=0)  # (T, H, W, C)
    gt_video = np.stack(gt_video, axis=0)      # (T, H, W, C)

    # Convert to torch tensors and add batch dimension
    pred_tensor = torch.from_numpy(pred_video).permute(0, 3, 1, 2).float().to(fid_metric.device)
    gt_tensor = torch.from_numpy(gt_video).permute(0, 3, 1, 2).float().to(fid_metric.device)

    # Update FID metric with the features from the current video pair
    fid_metric.update(pred_tensor, real=False)
    fid_metric.update(gt_tensor, real=True)

@torch.no_grad()
def fvd_update(pred_video, gt_video, fvd_metric):
    pred_video = np.stack(pred_video, axis=0)  # (T, H, W, C)
    gt_video = np.stack(gt_video, axis=0)      # (T, H, W, C)

    fvd_metric.add_fake_stats(pred_video[None, :])  # Add batch dimension
    fvd_metric.add_real_stats(gt_video[None, :])    # Add batch dimension

 
    
def compute_fid(fid_metric):
    # Compute FID score from the accumulated features
    return float(fid_metric.compute().item())

def compute_fvd(fvd_metric):
    return float(fvd_metric.compute_fvd_from_stats())


def align_hdr_pred_to_gt(pred, gt, percentile_low=10, percentile_high=90, eps=1e-8):
    """
    Align HDR prediction to GT using least-squares scale+bias (a, b),
    computed only on pixels whose GT luminance lies between given percentiles.

    Returns:
        aligned_pred, (a, b)
    """

    pred = np.clip(pred, 0, None)

    # Compute luminance mask from GT
    gt_lum = np.sum(gt, axis=2)
    p10, p90 = np.percentile(gt_lum, percentile_low), np.percentile(gt_lum, percentile_high)
    mask = (gt_lum >= p10) & (gt_lum <= p90)

    pred_masked = pred[mask]   # (N, 3)
    gt_masked   = gt[mask]     # (N, 3)

    # Flatten over channels
    x = pred_masked.reshape(-1)
    y = gt_masked.reshape(-1)

    # Solve min ||a*x + b - y||^2
    A = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)

    aligned_pred = pred * a + b
    #clamp aligned pred and gt to 1000 max
    aligned_pred = np.clip(aligned_pred, 0, 1000)
    gt_clamped = np.clip(gt, 0, 1000)
    return aligned_pred, gt_clamped, (a, b)

def pre_hdr_p3(img):
    eps = 1e-8
    flat = img.reshape(-1)

    # 0.1 and 99.9 percentiles
    p_low = np.percentile(flat, 0.1)
    p_high = np.percentile(flat, 99.9)

    c = 1.0
    d = 1000.0

    # linear mapping
    out = (img - p_low) * (d - c) / (p_high - p_low + eps) + c

    # clamp negative values
    out = np.maximum(out, 0)

    return out

def polyfit_ridge(x, y, deg, lam=1e-3, w0=None):
    # Build Vandermonde matrix
    V = np.vander(x, deg+1)

    # Prior center
    if w0 is None:
        w0 = np.zeros(deg+1)
        #make linear term 1
        w0[-2] = 1.0

    # Solve (V^T V + lam I) w = V^T y + lam w0
    A = V.T @ V + lam * np.eye(deg+1)
    b = V.T @ y + lam * w0
    w = np.linalg.solve(A, b)

    return w

import numpy as np
import cv2

def downsample_inter_area(img, factor):
    """
    Downsample using cv2.INTER_AREA (proper averaging).
    """
    if factor <= 1:
        return img
    H, W = img.shape[:2]
    return cv2.resize(
        img,
        (W // factor, H // factor),
        interpolation=cv2.INTER_AREA,
    )

def fit_and_apply_crf(
    pred,
    gt,
    deg=3,
    down_factor=4,
    p_low=10,
    p_high=90,
):
    """
    Fit a per-channel polynomial CRF in log space (pred → gt) using
    only values where gt lies between chosen percentiles.
    Applies the CRF to the full-resolution pred.
    """
    eps = 1
    H, W, C = pred.shape
    assert C == 3

    # --- Downsample for fitting using INTER_AREA ---
    pred_fit = downsample_inter_area(pred, down_factor)
    gt_fit   = downsample_inter_area(gt,   down_factor)

    # --- Flatten ---
    pred_f = pred_fit.reshape(-1, 3)
    gt_f   = gt_fit.reshape(-1, 3)
    num_pixels = pred_f.shape[0] * pred_f.shape[1]

    # --- Compute percentiles per channel on gt ---
    p_lo_vals = np.percentile(gt_f, p_low,  axis=0)
    p_hi_vals = np.percentile(gt_f, p_high, axis=0)

    # --- Valid mask: positive + percentile range ---
    valid_positive = (pred_f > 0).all(axis=1) & (gt_f > 0).all(axis=1)
    valid_range = np.ones(len(gt_f), dtype=bool)
    for c in range(3):
        valid_range &= (gt_f[:, c] >= p_lo_vals[c]) & (gt_f[:, c] <= p_hi_vals[c])

    valid = valid_positive & valid_range

    pred_f = pred_f[valid]
    gt_f   = gt_f[valid]

    # --- Log regression ---
    log_pred = np.log(pred_f + eps)
    log_gt   = np.log(gt_f   + eps)

    coeffs = np.zeros((3, deg + 1), dtype=np.float32)
    for c in range(3):
        x = log_pred[:, c]
        y = log_gt[:, c]
        coeffs[c] = polyfit_ridge(
            x, y, deg,
            lam=0.1 * num_pixels / (deg + 1)
        )

    # --- Apply CRF to full-res pred ---
    pred_safe = np.clip(pred, eps, None)
    log_pred_full = np.log(pred_safe)

    corrected = np.zeros_like(pred_safe)
    for c in range(3):
        p = coeffs[c]
        log_corr = np.polyval(p, log_pred_full[..., c])
        corrected[..., c] = np.exp(log_corr)

    return corrected, coeffs

