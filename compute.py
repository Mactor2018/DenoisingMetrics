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

# # 抑制 torchvision 中关于 pretrained 参数的弃用警告
# warnings.filterwarnings('ignore', category=UserWarning, message='.*pretrained.*')
# warnings.filterwarnings('ignore', category=UserWarning, message='.*weights.*')

class NotSameNameError(Exception):
    pass

# LPIPS 函数
def compute_lpips(img1, img2):
    # images should be normalized to [-1, 1] and have the same shape [batch, 3, H, W]
    # 将 numpy 数组转换为 tensor
    img1 = torch.from_numpy(img1).float() / 255.0 * 2 - 1
    img2 = torch.from_numpy(img2).float() / 255.0 * 2 - 1
    # 转换形状从 HWC 到 CHW，并添加 batch 维度
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    return lpips_model(img1, img2).item()

# PSNR函数
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
"""
nohup python -u compute.py > compute.out 2>&1 &

"""
# SSIM函数
def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

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
    files1 = sorted([f for f in all_files1 if f.endswith(('.png', '.jpg', '.bmp', '.jpeg'))])
    files2 = sorted([f for f in all_files2 if f.endswith(('.png', '.jpg', '.bmp', '.jpeg'))])

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
        if file1.endswith(('.png', '.jpg', '.bmp')) and file2.endswith(('.png', '.jpg', '.bmp')):
            img1 = cv2.imread(os.path.join(folder1, file1))
            img2 = cv2.imread(os.path.join(folder2, file2))

            # 检查图像是否具有相同的尺寸，如果不同则调整大小
            if img1.shape != img2.shape:
                print(f"[Warning] {file1} and {file2} have different shapes")

                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

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
    
    # 计算两个文件夹的FID
    if fid_gt_cache:
        avg_fid = fid_score.calculate_fid_given_paths(
            paths=[folder1, fid_gt_cache],
            batch_size=64,
            device=args.device,
            dims=2048,
            num_workers=0,  # Windows 上设置为 0 避免多进程问题
        )
    else:
        avg_fid = fid_score.calculate_fid_given_paths(
            paths=[folder1, folder2],
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
