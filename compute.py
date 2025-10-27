import cv2
import os
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



class NotSameNameError(Exception):
    pass

def _preprocess_to_256(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == 256 and w == 256:
        return img
    interp = cv2.INTER_AREA if (h >= 256 and w >= 256) else cv2.INTER_CUBIC
    return cv2.resize(img, (256, 256), interpolation=interp)

# LPIPS 函数（输入为 BGR，内部转 RGB）
def compute_lpips(img1_bgr, img2_bgr):
    device = next(lpips_model.parameters()).device if hasattr(lpips_model, 'parameters') else 'cpu'
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    img1_tensor = torch.from_numpy(img1_rgb).float() / 255.0 * 2 - 1
    img2_tensor = torch.from_numpy(img2_rgb).float() / 255.0 * 2 - 1
    img1_tensor = img1_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    img2_tensor = img2_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return lpips_model(img1_tensor, img2_tensor).item()

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
{'-'*20}
平均FID: {avg_fid:.4f}
平均PSNR: {avg_psnr:.4f}
平均SSIM: {avg_ssim:.4f}
平均LPIPS: {avg_lpips:.4f}
{'='*20}
    """
    print(s)


# 函数：遍历文件夹并计算指标
def compare_folders(args):
    folder1 = args.pred_dir
    folder2 = args.gt_dir
    fid_gt_cache = args.fid_gt_cache

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
    

    # 预处理临时目录（用于 FID）
    tmp_root = os.path.join(os.path.dirname(folder1), "_tmp_fid")
    tmp_pred_dir = os.path.join(tmp_root, "pred")
    tmp_gt_dir = os.path.join(tmp_root, "gt")
    if os.path.exists(tmp_pred_dir):
        shutil.rmtree(tmp_pred_dir)
    if os.path.exists(tmp_gt_dir):
        shutil.rmtree(tmp_gt_dir)
    os.makedirs(tmp_pred_dir, exist_ok=True)
    os.makedirs(tmp_gt_dir, exist_ok=True)

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

            # 统一预处理到 256x256
            img1 = _preprocess_to_256(img1)
            img2 = _preprocess_to_256(img2)
            if img1 is None or img2 is None:
                print(f"预处理失败: {file1} 或 {file2}")
                continue
            # 为 FID 保存预处理图像
            cv2.imwrite(os.path.join(tmp_pred_dir, file1), img1)
            cv2.imwrite(os.path.join(tmp_gt_dir, file2), img2)

            # 计算PSNR和SSIM
            psnr_value = psnr(img1, img2)
            ssim_value = ssim(img1, img2)
            lpips_value = compute_lpips(img1, img2)

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
        # print(f"平均PSNR: {avg_psnr:.4f}")
        # print(f"平均SSIM: {avg_ssim:.4f}")
        # print(f"平均LPIPS: {avg_lpips:.4f}")
    else:
        print("没有找到有效的图像进行计算")
        return
    
    # 计算两个文件夹的 FID（基于预处理后的临时目录），忽略 .npz 缓存
    if len(os.listdir(tmp_pred_dir)) == 0 or len(os.listdir(tmp_gt_dir)) == 0:
        print("预处理临时目录为空，无法计算 FID")
        avg_fid = float('nan')
    else:
        if fid_gt_cache and os.path.exists(fid_gt_cache) and fid_gt_cache.endswith('.npz'):
            print("检测到 FID 缓存，但已忽略，因为图像被统一缩放到 256x256")
        avg_fid = fid_score.calculate_fid_given_paths(
            paths=[tmp_pred_dir, tmp_gt_dir],
            batch_size=64,
            device=args.device,
            dims=2048,
            num_workers=0,  # Windows 上设置为 0 避免多进程问题
        )
    # print(f"平均FID: {avg_fid:.4f}")
    
    # 输出最终报告
    finalReport(avg_fid, avg_psnr, avg_ssim, avg_lpips, args)
    


def getArgs():
    parser = argparse.ArgumentParser(description='Compute PSNR/SSIM between two folders')
    parser.add_argument('--pred_dir','-p', required=True, type=str, help='Directory of predicted images')
    parser.add_argument('--gt_dir','-g', required=True, type=str, help='Directory of ground truth images')
    parser.add_argument('--device', default='cpu', type=str, help='Device to use for calculation')
    # LPIPS Model
    parser.add_argument('--lpips_model', default='alex', type=str, help='LPIPS Model: alex or vgg')
    # FID GT .npz Cache
    parser.add_argument('--fid_gt_cache', default=None, type=str, help='FID GT .npz Cache')
    # Strict Check
    parser.add_argument('--strict_check','-s', action='store_true', help='Strict Check: Check if the file names in the two folders are the same')
    return parser.parse_args()

if __name__ == '__main__':
    args = getArgs()

    # Strict Check
    if args.strict_check:
        print("[INFO] Strict Check Enabled")
    # Load LPIPS 
    lpips_model = lpips.LPIPS(net=args.lpips_model)
    print(f'LPIPS {args.lpips_model} has been initialized')
    
    # device = args.device
    print(args.pred_dir)
    print(args.gt_dir)
    
    compare_folders(args)
