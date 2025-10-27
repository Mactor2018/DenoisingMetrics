import argparse
import logging
from tqdm import tqdm
import os 
import cv2
from pytorch_fid import fid_score
from unref import getUIQM, getUCIQE
import json
from pathlib import Path
from compute import NotSameNameError
import lpips
import torch
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import shutil

# Global LPIPS model
lpips_model = None

# logging config
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] (%(levelname)s) %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 先只添加控制台处理器，文件处理器在 main() 中添加
logger.addHandler(logging.StreamHandler())


def getSortedImageFilePaths(folder:str)->list:
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# def getArgs():
#     parser = argparse.ArgumentParser(description='Compute Denoising Metrics: PSNR, SSIM, LPIPS, FID, UIQM, UCIQE')
#     parser.add_argument('--pred_dir','-p', required=True, type=str, help='Directory of predicted images')
#     parser.add_argument('--gt_dir','-g', required=False, type=str, help='Directory of ground truth images. If not provided, only UIQM and UCIQE will be computed')
#     parser.add_argument('--device', default='cpu', type=str, help='Device to use for calculation')
#     parser.add_argument('--all_metrics','-a', action='store_true', default=False, help='Compute all metrics including UIQM, UCIQE if GT provided')
#     parser.add_argument('--output_dir','-o', required=True, type=str, help='Directory to save the results: metrics_per_image and avg_metrics')
#     # LPIPS Model
#     # parser.add_argument('--lpips_model', default='alex', type=str, help='LPIPS Model: alex or vgg')
#     parser.add_argument('--lpips_model', choices=['alex', 'vgg'], default='alex',type=str, help='LPIPS Model: alex or vgg')
#     # FID GT .npz Cache
#     parser.add_argument('--fid_gt_cache', default=None, type=str, help='FID GT .npz Cache')
#     # Strict Check
#     parser.add_argument('--strict_check','-s', action='store_true', help='Strict Check: Check if the file names in the two folders are the same')
    
#     return parser.parse_args()

def parse_config(config_file:str)->dict:
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config
    except Exception as e:
        logger.error(f"Failed to parse config file: {e}")
        return None
    

def _check(parentFolder, value) -> bool:
    if not value: 
        return False
    return Path(parentFolder, value).exists()

def checkConfigPath(config:dict)->bool:
    dataset_config = config.get('datasets',{})
    if not dataset_config:
        logger.error("Dataset config not found in config file")
        return False

    for dataset_name, dataset_info in tqdm(dataset_config.items(), desc="Checking dataset paths", total=len(dataset_config)):
        parent_folder = dataset_info.get('parentFolder',None)
        if not parent_folder:
            logger.error(f"Parent folder not found for dataset: {dataset_name}")
            return False
        elif not os.path.exists(parent_folder):
            logger.error(f"Parent folder not found for dataset: {dataset_name}")
            return False
        else:
            # logger.info(f"Parent folder found for dataset: {dataset_name}")
            # check the gt_dir
            gt_dir = dataset_info.get('gt_dir',None)
            if not _check(parent_folder, gt_dir):
                    logger.warning(f"GT directory not found or not exists for dataset: {dataset_name}. Only UIQM and UCIQE will be computed")
            # check the fid_gt_cache
            fid_gt_cache = dataset_info.get('fid_gt_cache',None)
            if not _check(parent_folder, fid_gt_cache):
                logger.warning(f"FID GT cache not found or not exists for dataset: {dataset_name}.")
            # check the pred_dir
            pred_dir = dataset_info.get('pred_dir',None)
            if not pred_dir:
                logger.error(f"Pred directory not provided for dataset: {dataset_name}")
                return False
            else:
                for m_name, m_dir in tqdm(pred_dir.items(), desc=f"Checking {dataset_name} methods", total=len(pred_dir)):
                    if not _check(parent_folder, m_dir):
                        err_msg = f"Pred directory not found or not exists: {parent_folder}/{m_dir}"
                        logger.error(err_msg)
                        return False
    return True


def _preprocess_to_256(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == 256 and w == 256:
        return img
    interp = cv2.INTER_AREA if (h >= 256 and w >= 256) else cv2.INTER_CUBIC
    return cv2.resize(img, (256, 256), interpolation=interp)

def compute_lpips_local(img1_bgr, img2_bgr):
    """使用全局 lpips_model 计算 LPIPS（RGB，[-1,1]，NCHW）"""
    device = next(lpips_model.parameters()).device
    # BGR -> RGB
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    # [0,255] -> [-1,1]
    img1_tensor = torch.from_numpy(img1_rgb).float() / 255.0 * 2 - 1
    img2_tensor = torch.from_numpy(img2_rgb).float() / 255.0 * 2 - 1
    # HWC -> NCHW
    img1_tensor = img1_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    img2_tensor = img2_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return lpips_model(img1_tensor, img2_tensor).item()

def compare_folders(folder1, folder2, fid_gt_cache, m_name, device='cpu')->dict:
    # calculate PSNR, SSIM, LPIPS, FID

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
        # print("文件夹中的文件数量不一致")
        # print("files1: ", files1)
        # print("files2: ", files2)
        logger.error(f"The number of files in the two folders is not the same: {folder1} and {folder2}")
        return
    

    # 预处理临时目录（用于 FID）
    tmp_root = os.path.join(os.path.dirname(folder1), "_tmp_fid")
    tmp_pred_dir = os.path.join(tmp_root, m_name, "pred")
    tmp_gt_dir = os.path.join(tmp_root, m_name, "gt")
    if os.path.exists(tmp_pred_dir):
        shutil.rmtree(tmp_pred_dir)
    if os.path.exists(tmp_gt_dir):
        shutil.rmtree(tmp_gt_dir)
    os.makedirs(tmp_pred_dir, exist_ok=True)
    os.makedirs(tmp_gt_dir, exist_ok=True)

    # 遍历两个文件夹的同名文件
    for file1, file2 in tqdm(zip(files1, files2), total=n1):
        fname1, fname2 = os.path.splitext(file1), os.path.splitext(file2)
        if fname1 != fname2:
            raise NotSameNameError(f"{file1} and {file2} have different names! This may influence the evaluation results!")
        
        # 只处理图片文件
        if file1.lower().endswith(('.png', '.jpg', '.jpeg')) and file2.lower().endswith(('.png', '.jpg', '.jpeg')):
            img1 = cv2.imread(os.path.join(folder1, file1))
            img2 = cv2.imread(os.path.join(folder2, file2))
            
            # 检查图像是否成功读取
            if img1 is None or img2 is None:
                logger.warning(f"Failed to read images: {file1} or {file2}, skipping...")
                continue

            # 统一预处理到 256x256
            img1 = _preprocess_to_256(img1)
            img2 = _preprocess_to_256(img2)
            if img1 is None or img2 is None:
                logger.warning(f"Preprocess failed for: {file1} or {file2}, skipping...")
                continue
            # 为 FID 存储预处理图像
            cv2.imwrite(os.path.join(tmp_pred_dir, file1), img1)
            cv2.imwrite(os.path.join(tmp_gt_dir, file2), img2)

            # 计算PSNR和SSIM（使用 skimage）
            # skimage 的 psnr 需要指定 data_range（图像的值域范围）
            psnr_value = psnr(img1, img2, data_range=255)
            # skimage 的 ssim 需要指定 channel_axis 和 data_range
            ssim_value = ssim(img1, img2, channel_axis=2, data_range=255)
            lpips_value = compute_lpips_local(img1, img2)

            tqdm.write(f"{file1} {file2} PSNR: {psnr_value} SSIM: {ssim_value} LPIPS: {lpips_value}")
            # 累加到总值
            psnr_total += psnr_value
            ssim_total += ssim_value
            lpips_total += lpips_value
            count += 1
    # logger.info(f"The number of images processed: {count}")
    # 计算平均值
    if count > 0:
        avg_psnr = psnr_total / count
        avg_ssim = ssim_total / count
        avg_lpips = lpips_total / count
    else:
        logger.error("No valid images found for calculation")
        return
    
    # 计算两个文件夹的 FID（基于预处理后的临时目录），忽略 .npz 缓存以避免尺寸不一致
    avg_fid = None
    if os.path.isdir(tmp_pred_dir) and os.path.isdir(tmp_gt_dir):
        if len(os.listdir(tmp_pred_dir)) == 0 or len(os.listdir(tmp_gt_dir)) == 0:
            logger.error("Temporary directories for FID are empty, cannot compute FID")
        else:
            if fid_gt_cache and os.path.exists(fid_gt_cache) and fid_gt_cache.endswith('.npz'):
                logger.warning("Ignoring provided FID cache (.npz) because images were resized to 256x256.")
            avg_fid = fid_score.calculate_fid_given_paths(
                paths=[tmp_pred_dir, tmp_gt_dir],
                batch_size=64,
                device=device,
                dims=2048,
                num_workers=0,  # Windows 上设置为 0 避免多进程问题
            )
    else:
        logger.warning("No temporary directories provided for FID; skipping FID computation.")
    return {
        'method': m_name,
        'count': count,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips,
        'fid': avg_fid
    }
    

def metrics_with_ref(cfg:dict, device='cpu')->dict:
    # calculate PSNR, SSIM, LPIPS, FID
    results = {}
    parentFolder = cfg['parentFolder']
    gtDir = cfg['gt_dir']
    _fid_gt_cache = cfg.get('fid_gt_cache',None)
    fid_gt_cache = os.path.join(parentFolder,_fid_gt_cache if _fid_gt_cache else gtDir)
    predDirList = cfg['pred_dir']

    for m_name, m_dir in predDirList.items():
        pred_dir = os.path.join(parentFolder, m_dir) # pred
        gt_dir = os.path.join(parentFolder, gtDir) # gt
        method_result = compare_folders(pred_dir, gt_dir, fid_gt_cache, m_name, device) # psnr, ssim, lpips, fid
        if method_result:
            results[m_name] = method_result
            # logger.info(f"Method {m_name}: {method_result}")

    return results

def _compute_uiqm_uciqe(folder:str)->tuple:
    files = getSortedImageFilePaths(folder)
    for f in tqdm(files, desc=f"Computing UIQM and UCIQE for {folder}", total=len(files)):
        img_BGR = cv2.imread(f)
        if img_BGR is None:
            logger.warning(f"Failed to read image: {f}, skipping...")
            continue
        uiqm = getUIQM(img_BGR)
        uciqe = getUCIQE(img_BGR)
        yield uiqm, uciqe


def metrics_without_ref(cfg:dict)->dict:
    # calculate UIQM, UCIQE
    results = {}
    parentFolder = cfg['parentFolder']
    predDirList = cfg['pred_dir']
    for m_name, m_dir in predDirList.items():
        pred_dir = os.path.join(parentFolder, m_dir) # pred dir
        uiqm_list, uciqe_list = zip(*_compute_uiqm_uciqe(pred_dir))
        if len(uiqm_list) > 0 and len(uciqe_list) > 0:
            method_result = {
                'method': m_name,
                'uiqm': sum(uiqm_list) / len(uiqm_list),
                'uciqe': sum(uciqe_list) / len(uciqe_list)
            }
            results[m_name] = method_result
            logger.info(f"Method {m_name}: {method_result}")
        else:
            raise ValueError(f"No valid images found for calculation: {pred_dir}")

    return results

def saveResults(results:dict, output_path:str, dataset_name:str):
    """保存结果到 CSV 文件"""
    # 先创建目录
    os.makedirs(output_path, exist_ok=True)
    
    # 将结果转换为适合 DataFrame 的格式
    df = pd.DataFrame(results).T  # 转置，使方法名作为行索引
    df.index.name = 'Method'
    
    # 保存为 CSV
    csv_path = os.path.join(output_path, f"{dataset_name}_results.csv")
    df.to_csv(csv_path)
    logger.info(f"Results saved to: {csv_path}")


def main():
    global lpips_model
    
    parser = argparse.ArgumentParser(description='Compute Denoising Metrics: PSNR, SSIM, LPIPS, FID, UIQM, UCIQE')
    parser.add_argument('--config','-c', required=True, type=str, help='Config file')
    args = parser.parse_args()
    
    # 解析配置文件
    config = parse_config(args.config)
    if config is None:
        logger.error("Failed to load config file. Exiting...")
        return
    
    # 获取输出路径并创建目录
    output_path = config.get("output_path", "./metrics")
    os.makedirs(output_path, exist_ok=True)
    
    # 添加文件日志处理器到指定目录（只添加一个主日志文件）
    file_handler = logging.FileHandler(os.path.join(output_path, 'metrics.log'))
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] (%(levelname)s) %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Output directory: {output_path}")
    
    # 初始化 LPIPS 模型
    cfg_device = config.get('device', 'cpu')
    device = 'cuda' if (cfg_device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    lpips_model_name = config.get('lpips_model', 'alex')
    lpips_model = lpips.LPIPS(net=lpips_model_name).to(device)
    logger.info(f'LPIPS model ({lpips_model_name}) has been initialized on device: {device}')
    
    # 检查配置路径
    if checkConfigPath(config):
        logger.info("Paths checked successfully")
    else:
        raise ValueError("Paths checked failed")
    
    # 处理每个数据集
    for dataset_name, dataset_cfg in config['datasets'].items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*50}")
        
        gt_dir = dataset_cfg.get('gt_dir',None)
        all_results = {}
        
        if gt_dir:
            # 有参考图像：计算 PSNR, SSIM, LPIPS, FID
            ref_results = metrics_with_ref(dataset_cfg, device)
            all_results.update(ref_results)
            
            # 如果配置要求计算所有指标，还要计算无参考指标
            if config.get("all_metrics",False):
                logger.info("Computing unreferenced metrics (UIQM, UCIQE)...")
                unref_results = metrics_without_ref(dataset_cfg)
                # 合并结果
                for method_name in all_results:
                    if method_name in unref_results:
                        all_results[method_name].update(unref_results[method_name])
        else:
            # 无参考图像：只计算 UIQM, UCIQE
            logger.info("No GT directory provided, only computing unreferenced metrics...")
            all_results = metrics_without_ref(dataset_cfg)
        
        # 输出最终结果
        logger.info(f"\n{'='*50}")
        logger.info(f"Final Results for {dataset_name}:")
        logger.info(f"{'='*50}")
        for method_name, metrics in all_results.items():
            logger.info(f"{method_name}: {metrics}")
        logger.info(f"{'='*50}\n")

        # 保存结果到 CSV 文件
        saveResults(all_results, output_path, dataset_name)


    
if __name__ == "__main__":
    main()