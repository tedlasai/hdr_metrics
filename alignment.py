import numpy as np
import colour
from colour.models import RGB_COLOURSPACE_BT709, eotf_inverse_ST2084, eotf_ST2084


def ptf(x, ptf_type="pq", forward=True):
    """
    Per-pixel non-linear transform using `colour`.

    Parameters
    ----------
    x : ndarray
        Absolute luminance (cd/m^2) or arbitrary positive values if you
        approximate calibration.
    ptf_type : {'pq', 'log', 'linear'}
    forward : bool
        True: encode, False: decode.
    """
    x = np.asarray(x, dtype=np.float64)

    if ptf_type.lower() == "pq":
        # ST2084 expects absolute luminance in cd/m^2, clipped at 10000.
        if forward:
            return eotf_inverse_ST2084(np.clip(x, 0.0, 10000.0))
        else:
            return eotf_ST2084(np.clip(x, 0.0, 1.0))

    if ptf_type.lower() == "log":
        return np.log10(x) if forward else np.power(10.0, x)

    # 'linear' / anything else: identity
    return x

CS_BT709 = RGB_COLOURSPACE_BT709
WP = CS_BT709.whitepoint  # D65

def rgb_to_luv_bt709(rgb):
    """
    Convert BT.709 linear RGB (H, W, 3) -> CIE L*u*v* (same shape).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    h, w, c = rgb.shape
    assert c == 3

    rgb_flat = rgb.reshape(-1, 3)

    XYZ = colour.RGB_to_XYZ(
        rgb_flat,
        CS_BT709.whitepoint,
        CS_BT709.whitepoint,
        CS_BT709.matrix_RGB_to_XYZ,
    )
    Luv = colour.XYZ_to_Luv(XYZ, CS_BT709.whitepoint)

    return Luv.reshape(h, w, 3)


def luv_to_rgb_bt709(luv):
    """
    Convert CIE L*u*v* (H, W, 3) -> BT.709 linear RGB (same shape).
    """
    luv = np.asarray(luv, dtype=np.float64)
    h, w, c = luv.shape
    assert c == 3

    luv_flat = luv.reshape(-1, 3)

    XYZ = colour.Luv_to_XYZ(luv_flat, CS_BT709.whitepoint)
    rgb = colour.XYZ_to_RGB(
        XYZ,
        CS_BT709.whitepoint,
        CS_BT709.whitepoint,
        CS_BT709.matrix_XYZ_to_RGB,
    )

    rgb = rgb.reshape(h, w, 3)
    return np.clip(rgb, 0.0, None)


def crf_correction_luv_bt709(
    Ir,
    Igt,
    deg=3,
    lam=0.01,
    ptf_type="pq",
    normalize=False,
    L_min=5e-3,
    sc=500.0,
):
    """
    CRF correction in CIE Luv, assuming BT.709 linear RGB input.

    Parameters
    ----------
    Ir : ndarray, (H, W, 3)
        Input HDR image (BT.709, linear).
    Igt : ndarray, (H, W, 3)
        Reference HDR image (BT.709, linear).
    deg : int
        Polynomial degree.
    lam : float
        Regularization strength for UV.
    ptf_type : {'pq', 'log', 'linear'}
        Non-linear transform type for L channel.
    normalize : bool
        If True, do approximate scaling to absolute luminance
        (anchor median to `sc`).
    L_min : float
        Minimum luminance clamp.
    sc : float
        Target median luminance for approximate calibration.
    """
    Ir = np.asarray(Ir, dtype=np.float64)
    Igt = np.asarray(Igt, dtype=np.float64)

    # Approximate calibration (same logic as MATLAB)
    if normalize:
        scale_gt = np.median(Igt)
        Igt = sc * Igt / scale_gt
        Ir = sc * Ir / np.median(Ir)

    Igt = np.clip(Igt, L_min, None)
    Ir = np.clip(Ir, L_min, None)

    # ---- RGB -> Luv (BT.709) ----
    Ir_luv = rgb_to_luv_bt709(Ir)
    Igt_luv = rgb_to_luv_bt709(Igt)

    # Split channels
    Ir_L = Ir_luv[..., 0:1]        # keep as (H,W,1) so corr_opt sees "1-channel image"
    Ir_uv = Ir_luv[..., 1:3]
    Igt_L = Igt_luv[..., 0:1]
    Igt_uv = Igt_luv[..., 1:3]

    # ---- L channel: no regularization, chosen ptf_type ----
    params_L = {
        "deg": deg,
        "lambda": 0.0,       # no regularization for luminance
        "ptf_type": ptf_type,
        "L_min": L_min,
    }
    It_L, W_L = corr_opt(Ir_L, Igt_L, params_L)

    # ---- UV channels: linear domain, with regularization ----
    params_uv = {
        "deg": deg,
        "lambda": lam,
        "ptf_type": "linear",  # same as your 'lin' branch
        "L_min": L_min,
    }
    It_uv, W_uv = corr_opt(Ir_uv, Igt_uv, params_uv)

    # Pack back into Luv image
    It_luv = np.zeros_like(Ir_luv)
    It_luv[..., 0] = It_L[..., 0]
    It_luv[..., 1:3] = It_uv

    # Luv -> RGB (BT.709)
    It = luv_to_rgb_bt709(It_luv)

    # Undo normalization if needed
    if normalize:
        It = scale_gt * It / sc

    # weights: same structure as MATLAB: {W_L, W_uv}
    W = {"L": W_L, "uv": W_uv}
    return It, W
import numpy as np

def lin_matrix(I, deg):
    """
    Build polynomial design matrix with optional cross-channel terms.
    I: (H, W, C)
    Returns M: (N, F) with N = H*W.
    """
    H, W, C = I.shape
    N = H * W

    M_ = I.reshape(N, C)  # base features per pixel
    cols = []

    # Cross-channel products (if multi-channel)
    if C > 1:
        for i in range(C - 1):
            for j in range(i + 1, C):
                cols.append(M_[:, i] * M_[:, j])   # (N,)

    # Linear terms + constant
    cols.append(M_)                 # (N, C)
    cols.append(np.ones((N, 1)))    # (N, 1)

    # Concatenate all columns
    M = np.concatenate(
        [c if c.ndim == 2 else c[:, None] for c in cols],
        axis=1,
    )

    # Higher-order terms: prepend M_**d like in MATLAB
    for d in range(2, deg + 1):
        M = np.concatenate([M_ ** d, M], axis=1)

    return M


def corr_opt(Ir, Igt, params):
    """
    Polynomial CRF correction with Tikhonov regularization.

    Ir, Igt: image tensors, shape (H,W) or (H,W,C)

    params must contain:
      - 'deg'       : polynomial degree
      - 'lambda'    : regularization strength
      - 'ptf_type'  : 'pq', 'log', or 'linear'
      - 'L_min'     : minimum clamp
    """
    ptf_type = params["ptf_type"]
    L_min    = params.get("L_min", 5e-3)
    deg      = params["deg"]
    lam      = params["lambda"]

    # --- normalize shapes to (H, W, C) ---
    x = np.asarray(Ir, dtype=np.float64)
    y = np.asarray(Igt, dtype=np.float64)

    if x.ndim == 2:
        x = x[..., None]   # (H,W) -> (H,W,1)
    if y.ndim == 2:
        y = y[..., None]

    H, W, C = x.shape

    # --- forward non-linear transform ---
    y_t = ptf(y, ptf_type, forward=True)
    x_t = ptf(x, ptf_type, forward=True)

    # --- matrix formulation ---
    N = H * W
    Y = y_t.reshape(N, C)
    X = lin_matrix(x_t, deg)  # (N, F)
    F = X.shape[1]

    # Identity mapping (for regularization) on last C "linear" coeffs
    W0 = np.zeros((F, C))
    # Mimic MATLAB: W0(end-zz:end-1,:) = eye(zz);
    # in 0-based Python this is F-C-1 .. F-2 inclusive.
    start = F - C - 1
    end   = F - 1
    W0[start:end, :] = np.eye(C)

    # --- regularized least squares ---
    sc = lam * N / F
    A = X.T @ X + sc * np.eye(F)
    B = X.T @ Y + sc * W0
    W_out = np.linalg.solve(A, B)   # (F, C)

    # --- apply correction ---
    print(H, W, C)
    It_t = (X @ W_out).reshape(H, W, C)
    # clamp and inverse transform
    It_t = np.maximum(It_t, ptf(L_min, ptf_type, forward=True))
    It   = ptf(It_t, ptf_type, forward=False)

    return It, W_out
