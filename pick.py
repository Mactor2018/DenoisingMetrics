import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from unref import getUIQM, getUCIQE


ALLOWED_EXTS = (".png", ".jpg", ".jpeg")


def _preprocess_to_256(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    if h == 256 and w == 256:
        return img
    interp = cv2.INTER_AREA if (h >= 256 and w >= 256) else cv2.INTER_CUBIC
    return cv2.resize(img, (256, 256), interpolation=interp)


def _list_images_by_basename(directory: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not directory or not os.path.isdir(directory):
        return mapping
    for name in os.listdir(directory):
        if not name.lower().endswith(ALLOWED_EXTS):
            continue
        base = os.path.splitext(name)[0]
        mapping[base] = os.path.join(directory, name)
    return mapping


def _psnr_ssim(pred_img, gt_img) -> Tuple[float, float]:
    pred_img = _preprocess_to_256(pred_img)
    gt_img = _preprocess_to_256(gt_img)
    if pred_img is None or gt_img is None:
        raise ValueError("Image read/preprocess failed")
    psnr_val = sk_psnr(pred_img, gt_img, data_range=255)
    ssim_val = sk_ssim(pred_img, gt_img, channel_axis=2, data_range=255)
    return float(psnr_val), float(ssim_val)


def _ui_metrics(img) -> Tuple[float, float]:
    img = _preprocess_to_256(img)
    if img is None:
        raise ValueError("Image read/preprocess failed")
    # getUIQM/getUCIQE 接受 BGR
    uiqm = getUIQM(img)
    uciqe = getUCIQE(img)
    return float(uciqe), float(uiqm)


def _rank_index_desc(values: List[Tuple[str, float]]) -> Dict[str, int]:
    # values: [(method, val)], 高->低 排名（从1开始）
    sorted_vals = sorted(values, key=lambda x: x[1], reverse=True)
    ranks: Dict[str, int] = {}
    rank = 1
    for i, (m, v) in enumerate(sorted_vals):
        if i > 0 and v != sorted_vals[i - 1][1]:
            rank = i + 1
        ranks[m] = rank
    return ranks


def pick_with_ref(dataset_cfg: dict, top_k: int) -> List[str]:
    parent = dataset_cfg["parentFolder"]
    gt_dir_name = dataset_cfg.get("gt_dir")
    pred_dirs: Dict[str, str] = dataset_cfg.get("pred_dir", {})

    if "Ours" not in pred_dirs:
        print(f"[WARN] 数据集缺少 Ours：{parent}")
        return []

    gt_dir = os.path.join(parent, gt_dir_name) if gt_dir_name else None
    if not gt_dir or not os.path.isdir(gt_dir):
        print(f"[WARN] 无效 GT 目录：{gt_dir}")
        return []

    ours_dir = os.path.join(parent, pred_dirs["Ours"])
    if not os.path.isdir(ours_dir):
        print(f"[WARN] 无效 Ours 目录：{ours_dir}")
        return []

    # 准备方法目录（排除 GT），确保存在
    method_dirs: Dict[str, str] = {}
    for m_name, m_sub in pred_dirs.items():
        path = os.path.join(parent, m_sub)
        if os.path.isdir(path):
            method_dirs[m_name] = path

    gt_map = _list_images_by_basename(gt_dir)
    if not gt_map:
        print(f"[WARN] GT 无图像：{gt_dir}")
        return []

    # 以 Ours 的可用图为基准
    ours_map = _list_images_by_basename(ours_dir)
    bases = sorted(set(gt_map.keys()) & set(ours_map.keys()))
    if not bases:
        print(f"[WARN] Ours 与 GT 无共同图像：{parent}")
        return []

    best_candidates: List[Tuple[str, float, float, float]] = []
    # (base, mean_rank, ours_psnr, ours_ssim)

    for base in tqdm(bases, desc=f"[WithRef] {os.path.basename(parent)}"):
        gt_path = gt_map.get(base)
        if not gt_path:
            continue

        metric_per_method: Dict[str, Tuple[float, float]] = {}
        for m_name, m_dir in method_dirs.items():
            pred_path = _list_images_by_basename(m_dir).get(base)
            if not pred_path:
                continue
            pred_img = cv2.imread(pred_path)
            gt_img = cv2.imread(gt_path)
            if pred_img is None or gt_img is None:
                continue
            try:
                psnr_val, ssim_val = _psnr_ssim(pred_img, gt_img)
            except Exception:
                continue
            metric_per_method[m_name] = (psnr_val, ssim_val)

        if "Ours" not in metric_per_method:
            continue

        # 计算 Ours 在各指标上的名次与平均名次
        psnr_list = [(m, v[0]) for m, v in metric_per_method.items()]
        ssim_list = [(m, v[1]) for m, v in metric_per_method.items()]
        psnr_ranks = _rank_index_desc(psnr_list)
        ssim_ranks = _rank_index_desc(ssim_list)
        ours_rank_psnr = psnr_ranks["Ours"]
        ours_rank_ssim = ssim_ranks["Ours"]
        mean_rank = (ours_rank_psnr + ours_rank_ssim) / 2.0
        ours_psnr, ours_ssim = metric_per_method["Ours"]
        best_candidates.append((base, mean_rank, ours_psnr, ours_ssim))

    # 选出平均名次最靠前的 top_k，若相同则按 Ours PSNR 再按 SSIM 下降排序
    best_candidates.sort(key=lambda x: (x[1], -x[2], -x[3]))
    selected = [b for b, _, _, _ in best_candidates[:top_k]]
    return selected


def pick_no_ref(dataset_cfg: dict, top_k: int) -> List[str]:
    parent = dataset_cfg["parentFolder"]
    pred_dirs: Dict[str, str] = dataset_cfg.get("pred_dir", {})

    if "Ours" not in pred_dirs:
        print(f"[WARN] 数据集缺少 Ours：{parent}")
        return []

    ours_dir = os.path.join(parent, pred_dirs["Ours"])
    if not os.path.isdir(ours_dir):
        print(f"[WARN] 无效 Ours 目录：{ours_dir}")
        return []

    # 有效的方法目录
    method_dirs: Dict[str, str] = {}
    for m_name, m_sub in pred_dirs.items():
        path = os.path.join(parent, m_sub)
        if os.path.isdir(path):
            method_dirs[m_name] = path

    ours_map = _list_images_by_basename(ours_dir)
    if not ours_map:
        print(f"[WARN] Ours 无图像：{ours_dir}")
        return []

    bases = sorted(ours_map.keys())
    best_candidates: List[Tuple[str, float, float, float]] = []
    # (base, mean_rank, ours_uciqe, ours_uiqm)

    # 为避免重复列举每个方法的文件映射，先缓存
    per_method_maps: Dict[str, Dict[str, str]] = {m: _list_images_by_basename(d) for m, d in method_dirs.items()}

    for base in tqdm(bases, desc=f"[NoRef] {os.path.basename(parent)}"):
        metric_per_method: Dict[str, Tuple[float, float]] = {}
        for m_name, m_map in per_method_maps.items():
            img_path = m_map.get(base)
            if not img_path:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            try:
                uciqe_val, uiqm_val = _ui_metrics(img)
            except Exception:
                continue
            metric_per_method[m_name] = (uciqe_val, uiqm_val)

        if "Ours" not in metric_per_method:
            continue

        uciqe_list = [(m, v[0]) for m, v in metric_per_method.items()]
        uiqm_list = [(m, v[1]) for m, v in metric_per_method.items()]
        uciqe_ranks = _rank_index_desc(uciqe_list)
        uiqm_ranks = _rank_index_desc(uiqm_list)
        ours_rank_uciqe = uciqe_ranks["Ours"]
        ours_rank_uiqm = uiqm_ranks["Ours"]
        mean_rank = (ours_rank_uciqe + ours_rank_uiqm) / 2.0
        ours_uciqe, ours_uiqm = metric_per_method["Ours"]
        best_candidates.append((base, mean_rank, ours_uciqe, ours_uiqm))

    best_candidates.sort(key=lambda x: (x[1], -x[2], -x[3]))
    selected = [b for b, _, _, _ in best_candidates[:top_k]]
    return selected


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="挑选 Ours 方法在各指标下表现靠前的图片序号")
    parser.add_argument("--config", "-c", default="./config.json", help="配置文件路径")
    parser.add_argument("--top-k", type=int, default=10, help="每个数据集挑选的图片数量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    datasets = cfg.get("datasets", {})

    # UIEBD 与 LSUI：使用 PSNR/SSIM（有参考）
    for name in ["UIEBD", "LSUI"]:
        if name not in datasets:
            continue
        try:
            picks = pick_with_ref(datasets[name], args.top_k)
        except Exception as e:
            print(f"[ERROR] 处理 {name} 失败：{e}")
            picks = []
        print(f"{name}（PSNR+SSIM）挑选 Top{args.top_k}：")
        print(", ".join(picks) if picks else "(无)")
        print()

    # U45：使用 UCIQE/UIQM（无参考）
    if "U45" in datasets:
        try:
            picks = pick_no_ref(datasets["U45"], args.top_k)
        except Exception as e:
            print(f"[ERROR] 处理 U45 失败：{e}")
            picks = []
        print(f"U45（UCIQE+UIQM）挑选 Top{args.top_k}：")
        print(", ".join(picks) if picks else "(无)")
        print()


if __name__ == "__main__":
    main()



