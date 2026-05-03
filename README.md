# DenoisingMetrics

This is a codebase for evaluating Image Denoising and Enhancement tasks.

## Metrics
- **Reference-based**: PSNR, SSIM, LPIPS, FID
- **No-reference**: UIQM, UCIQE

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch`, `torchvision`
- `numpy`, `scipy`
- `opencv-python`
- `scikit-image`
- `tqdm`
- `lpips`
- `pytorch_fid`
- `threadpoolctl` (used to prevent system freeze during FID matrix computation)

## Usage

### Compute Reference-based Metrics
Run the following command to calculate PSNR, SSIM, LPIPS, and FID between two folders:

```bash
python compute.py -p /path/to/predicted_images -g /path/to/ground_truth_images
```

**Features**:
- **Auto-device detection**: Automatically uses CUDA if available, otherwise falls back to CPU.
- **Resource protection**: Limits BLAS threads during FID calculation to prevent system hangs.
- **Speed**: GPU-accelerated LPIPS and FID feature extraction.

### Parameters
- `--pred_dir`, `-p`: Directory of predicted images.
- `--gt_dir`, `-g`: Directory of ground truth images.
- `--device`: Device to use (auto/cpu/cuda). Default: `auto`.
- `--lpips_model`: LPIPS backbone (alex/vgg). Default: `alex`.
- `--strict_check`, `-s`: If set, ensures filenames match exactly.