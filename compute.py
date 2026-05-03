import os
# ── 限制 BLAS/OpenMP 线程数，防止 scipy.linalg.sqrtm 抢占全部 CPU 导致系统卡死 ──
# 必须在 numpy/scipy 导入之前设置
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import cv2
import numpy as np
import math
import torch
import warnings
# from torchvision import transforms
import argparse
from pytorch_fid import fid_score
import lpips
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import shutil
import gc
from scipy import linalg

try:
    from threadpoolctl import threadpool_limits
    HAS_THREADPOOLCTL = True
except ImportError:
    HAS_THREADPOOLCTL = False
    print("[WARN] threadpoolctl not found, install with: pip install threadpoolctl")
    print("[WARN] Falling back to env-var thread limiting only.")



class NotSameNameError(Exception):
    pass

def _preprocess_to_256(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == 256 and w == 256:
        return img
    else:
        raise ValueError(f"Image size must be 256x256, but got {h}x{w}")

# LPIPS 函数（输入为 BGR，内部转 RGB）
@torch.no_grad()
def compute_lpips(img1_bgr, img2_bgr, model, device):
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    img1_tensor = torch.from_numpy(img1_rgb).float().div_(255.0).mul_(2.0).sub_(1.0)
    img2_tensor = torch.from_numpy(img2_rgb).float().div_(255.0).mul_(2.0).sub_(1.0)
    img1_tensor = img1_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    img2_tensor = img2_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    val = model(img1_tensor, img2_tensor).item()
    # 及时释放 GPU 显存
    del img1_tensor, img2_tensor
    return val

# 使用 skimage 计算 PSNR
def psnr(img1, img2):
    return sk_psnr(img1, img2, data_range=255)
"""
nohup python -u compute.py > compute.out 2>&1 &

"""
# 使用 skimage 计算 SSIM（彩色）
def ssim(img1, img2):
    return sk_ssim(img1, img2, channel_axis=2, data_range=255)

def finalReport(avg_fid, avg_psnr, avg_ssim, avg_lpips, args):
    s = f"""
{'='*20}
[Final Report]
预测结果路径: {args.pred_dir}
真实结果路径: {args.gt_dir}
严格检查: {'开启' if args.strict_check else '关闭'}
LPIPS模型: {args.lpips_model}
FID GT Cache: {args.fid_gt_cache if args.fid_gt_cache else '无'}
计算设备: {args.device}
{'-'*20}
平均PSNR: {avg_psnr:.4f}
平均SSIM: {avg_ssim:.4f}
平均LPIPS: {avg_lpips:.4f}
平均FID: {avg_fid:.4f}
{'='*20}
    """
    print(s)

# ── 安全 FID 计算：分离特征提取与矩阵运算，限制 BLAS 线程 ──
def _compute_fid_safe(path1, path2, device, batch_size=16, dims=2048):
    """
    将 FID 拆成两步：
      1) Inception 特征提取（GPU 加速）
      2) Frechet Distance 矩阵运算（限制 BLAS 线程数，防止卡死）
    """
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import compute_statistics_of_path

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    print("[FID 1/3] Extracting features from pred_dir ...")
    m1, s1 = compute_statistics_of_path(path1, model, batch_size, dims, device, num_workers=0)

    print("[FID 2/3] Extracting features from gt_dir ...")
    m2, s2 = compute_statistics_of_path(path2, model, batch_size, dims, device, num_workers=0)

    # 释放 InceptionV3 显存
    del model
    torch.cuda.empty_cache()

    print("[FID 3/3] Computing Frechet Distance (sqrtm on 2048x2048 matrix) ...")
    # ★ 关键：限制 BLAS 线程数，防止 OpenBLAS 抢占全部 CPU 核心
    max_threads = 4
    if HAS_THREADPOOLCTL:
        with threadpool_limits(limits=max_threads, user_api='blas'):
            fid_val = _frechet_distance(m1, s1, m2, s2)
    else:
        fid_val = _frechet_distance(m1, s1, m2, s2)

    print(f"[FID] Done. FID = {fid_val:.4f}")
    return fid_val


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """与 pytorch_fid 完全一致的 Frechet Distance 实现"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


# 函数：遍历文件夹并计算指标
def compare_folders(args, lpips_model):
    folder1 = args.pred_dir
    folder2 = args.gt_dir
    fid_gt_cache = args.fid_gt_cache
    device = args.device

    psnr_total = 0
    ssim_total = 0
    lpips_total = 0
    count = 0

    # 可配置排序（此处使用升序便于与编号对齐）
    # 只获取图片文件
    all_files1 = os.listdir(folder1)
    all_files2 = os.listdir(folder2)
    files1 = sorted([f for f in all_files1 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    files2 = sorted([f for f in all_files2 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 确保两个文件夹的文件数相同
    n1, n2 = len(files1), len(files2)
    if n1 != n2:  
        print("文件夹中的文件数量不一致")
        print("files1: ", files1)
        print("files2: ", files2)
        return
    

    # 遍历两个文件夹的同名文件
    for file1, file2 in tqdm(zip(files1, files2), total=n1):
        if args.strict_check:
            fname1, fname2 = os.path.splitext(file1), os.path.splitext(file2)
            if fname1 != fname2:
                raise NotSameNameError(f"{file1} and {file2} have different names! This may influence the evaluation results!")

        # 只处理图片文件
        if file1.lower().endswith(('.png', '.jpg', '.jpeg')) and file2.lower().endswith(('.png', '.jpg', '.jpeg')):
            img1 = cv2.imread(os.path.join(folder1, file1))
            img2 = cv2.imread(os.path.join(folder2, file2))

            if img1 is None or img2 is None:
                print(f"读取图像失败: {file1} 或 {file2}")
                continue

            # 尺寸检查
            img1 = _preprocess_to_256(img1)
            img2 = _preprocess_to_256(img2)
            if img1 is None or img2 is None:
                print(f"尺寸检查失败: {file1} 或 {file2}")
                continue

            # 计算PSNR和SSIM
            psnr_value = psnr(img1, img2)
            ssim_value = ssim(img1, img2)
            lpips_value = compute_lpips(img1, img2, lpips_model, device)

            print(f"{file1} {file2} \n PSNR: {psnr_value} SSIM: {ssim_value} LPIPS: {lpips_value}")
            # 累加到总值
            psnr_total += psnr_value
            ssim_total += ssim_value
            lpips_total += lpips_value
            count += 1
    print(count)
    # 计算平均值
    if count > 0:
        avg_psnr = psnr_total / count
        avg_ssim = ssim_total / count
        avg_lpips = lpips_total / count
    else:
        print("没有找到有效的图像进行计算")
        return

    # 释放 LPIPS 模型显存，为 FID 腾出空间
    del lpips_model
    torch.cuda.empty_cache()
    gc.collect()

    # 计算两个文件夹的 FID，允许使用 FID GT Cache
    if len(os.listdir(folder1)) == 0 or len(os.listdir(folder2)) == 0:
        print("输入目录为空，无法计算 FID")
        avg_fid = float('nan')
    else:
        fid_target = args.fid_gt_cache if (args.fid_gt_cache and os.path.exists(args.fid_gt_cache) and args.fid_gt_cache.endswith('.npz')) else folder2
        avg_fid = _compute_fid_safe(folder1, fid_target, device)
    # 输出最终报告
    finalReport(avg_fid, avg_psnr, avg_ssim, avg_lpips, args)
    


def getArgs():
    parser = argparse.ArgumentParser(description='Compute PSNR/SSIM between two folders')
    parser.add_argument('--pred_dir','-p', required=True, type=str, help='Directory of predicted images')
    parser.add_argument('--gt_dir','-g', required=True, type=str, help='Directory of ground truth images')
    parser.add_argument('--device', default='auto', type=str, help='Device to use for calculation (auto/cpu/cuda)')
    # LPIPS Model
    parser.add_argument('--lpips_model', default='alex', type=str, help='LPIPS Model: alex or vgg')
    # FID GT .npz Cache
    parser.add_argument('--fid_gt_cache', default=None, type=str, help='FID GT .npz Cache')
    # Strict Check
    parser.add_argument('--strict_check','-s', action='store_true', help='Strict Check: Check if the file names in the two folders are the same')
    return parser.parse_args()

if __name__ == '__main__':
    args = getArgs()

    # 自动检测设备
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {args.device}")

    # Strict Check
    if args.strict_check:
        print("[INFO] Strict Check Enabled")
    # Load LPIPS 
    lpips_model = lpips.LPIPS(net=args.lpips_model).to(args.device).eval()
    print(f'LPIPS {args.lpips_model} has been initialized on {args.device}')
    
    # device = args.device
    print(args.pred_dir)
    print(args.gt_dir)
    
    compare_folders(args, lpips_model)
