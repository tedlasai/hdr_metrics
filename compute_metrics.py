#CONDA ENVIRONEMNT P312

import os
#add to system path
from metrics_helpers import read_image, pre_hdr_p3, align_hdr_pred_to_gt, psnr, vsi, piqe, lpips, hdr_vdp3, pu, reinhard_tonemap
import os
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from metrics_helpers import initialize_fid, initialize_fvd
from metrics_helpers import compute_fid, fid_update, compute_fvd, fvd_update


pred_dir = "/data2/saikiran.tedla/hdrvideo/diff/evaluations/ours_stuttgart/over/"
gt_dir =  "/data2/saikiran.tedla/hdrvideo/diff/evaluations/stuttgart/hdr/"
video_paths = sorted(os.listdir(pred_dir))

reinhard_fvd_metric = initialize_fvd()
reinhard_fid_metric = initialize_fid()

pu_fvd_metric = initialize_fvd()
pu_fid_metric = initialize_fid()

psnr_scores = []
vsi_scores = []
piqe_scores = []
lpips_scores = []

for video_path in video_paths:
    pred_video_dir = os.path.join(pred_dir, video_path)
    gt_video_dir = os.path.join(gt_dir, video_path)
    im_paths = sorted(os.listdir(pred_video_dir))[:16]

    reinhard_preds = []
    reinhard_gts = []
    pu_preds = []
    pu_gts = []
    for im_path in im_paths:
        pred_im_path = os.path.join(pred_video_dir, im_path)
        gt_im_path = os.path.join(gt_video_dir, im_path)
        cv2_hdr_pred = read_image(pred_im_path)
        cv2_hdr_gt = read_image(gt_im_path)
        cv2_hdr_gt = pre_hdr_p3(cv2_hdr_gt)
        cv2_hdr_pred,cv2_hdr_gt,_ = align_hdr_pred_to_gt(cv2_hdr_pred, cv2_hdr_gt)

        jod = hdr_vdp3(cv2_hdr_pred, cv2_hdr_gt)

        pu_pred, pu_gt = pu(cv2_hdr_pred), pu(cv2_hdr_gt)
        pu_preds.append(pu_pred/np.max(pu_gt))
        pu_gts.append(pu_gt/np.max(pu_gt))

        reinhard_pred, reinhard_gt = reinhard_tonemap(cv2_hdr_pred), reinhard_tonemap(cv2_hdr_gt)
        reinhard_preds.append(reinhard_pred)
        reinhard_gts.append(reinhard_gt)


        pu_psnr = psnr(pu_pred, pu_gt)
        psnr_scores.append(pu_psnr)
        print("PU-PSNR:", pu_psnr)

        pu_vsi = vsi(pu_pred, pu_gt)
        vsi_scores.append(pu_vsi)   
        print("PU-VSI:", pu_vsi)

        pu_piqe = piqe(pu_pred)
        piqe_scores.append(pu_piqe)
        print("PIQE Score:", pu_piqe)

        pu_lpips = lpips(reinhard_pred, reinhard_gt)
        lpips_scores.append(pu_lpips)
        print("LPIPS Score:", pu_lpips)


        fid_update(reinhard_pred, reinhard_gt, reinhard_fid_metric)
        fid_update(pu_pred, pu_gt, pu_fid_metric)
        
    fvd_update(reinhard_preds, reinhard_gts, reinhard_fvd_metric)
    fvd_update(pu_preds, pu_gts, pu_fvd_metric)

print("R-FID Score:", compute_fid(reinhard_fid_metric))
print("R-FVD Score:", compute_fvd(reinhard_fvd_metric))
print("PU-FID Score:", compute_fid(pu_fid_metric))
print("PU-FVD Score:", compute_fvd(pu_fvd_metric))

print("Average PU-PSNR:", np.mean(np.array(psnr_scores)))
print("Average PU-VSI:", np.mean(np.array(vsi_scores)))
print("Average PIQE:", np.mean(np.array(piqe_scores)))
print("Average LPIPS:", np.mean(np.array(lpips_scores)))

    #add fid and fvd (then loop over videos, computer, save to csv)


