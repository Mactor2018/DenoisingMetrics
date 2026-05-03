# DenoisingMetrics

This is a codebase for evaluating Image Denoising and Enhancement tasks, supporting both reference-based and no-reference metrics.

## Metrics
- **Reference-based**: PSNR, SSIM, LPIPS, FID
- **No-reference**: UIQM, UCIQE (Underwater Image Quality/Colorfulness Measures)

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```
Key dependencies: `torch`, `numpy`, `scipy`, `opencv-python`, `scikit-image`, `tqdm`, `lpips`, `pytorch_fid`, `threadpoolctl`.

---

## 1. Batch Evaluation (`main.py`)
Use `main.py` to evaluate multiple methods across multiple datasets at once using a configuration file.

### Usage
```bash
python main.py --config config.json
```

### Configuration (`config.json`)
The configuration file allows you to define global settings and dataset-specific paths:
- `device`: `cuda` or `cpu`.
- `all_metrics`: If `true`, computes UIQM and UCIQE even if Ground Truth is available.
- `output_path`: Directory where CSV results and logs will be saved.
- `datasets`: A dictionary where each key is a dataset name, containing:
    - `parentFolder`: Root path for the dataset.
    - `gt_dir`: Subfolder containing ground truth images.
    - `fid_gt_cache`: (Optional) Path to a `.npz` file containing precomputed FID statistics for the GT.
    - `pred_dir`: A dictionary mapping **Method Names** to their respective result subfolders.

---

## 2. Direct Folder Comparison (`compute.py`)
Use `compute.py` for a quick comparison between two specific folders (Predicted vs. Ground Truth).

### Usage
```bash
python compute.py -p /path/to/predicted_images -g /path/to/ground_truth_images
```

**Features**:
- **Auto-device detection**: Automatically uses CUDA if available.
- **Resource protection**: Limits BLAS threads during FID calculation to prevent system hangs.
- **Speed**: GPU-accelerated LPIPS and FID feature extraction.

---

## 3. No-Reference Evaluation (`unref.py`)
Use `unref.py` to calculate UIQM and UCIQE for a single folder of images without requiring ground truth.

### Usage
```bash
python unref.py --folder /path/to/images --output results.txt
```

---

## Parameters Summary

| Script | Parameter | Description |
| :--- | :--- | :--- |
| `main.py` | `--config`, `-c` | Path to the JSON configuration file. |
| `compute.py` | `--pred_dir`, `-p` | Directory of predicted images. |
| `compute.py` | `--gt_dir`, `-g` | Directory of ground truth images. |
| `compute.py` | `--device` | Device to use (`auto`/`cpu`/`cuda`). |
| `compute.py` | `--strict_check`, `-s` | Ensure filenames match exactly between folders. |
| `unref.py` | `--folder`, `-f` | Directory of images to evaluate. |
| `unref.py` | `--output`, `-o` | File path to save the per-image and average results. |