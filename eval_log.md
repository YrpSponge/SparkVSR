# SparkVSR Evaluation Log

> **Server**: ddcloud | **Path**: /data3/ryu455/SparkVSR | **Date**: 2026-05-01

---

## 1. Environment & Metrics Setup

| Component | Status | Notes |
|---|---|---|
| CLIPIQA | OK | pyiqa, fixed pkg_resources |
| MUSIQ | OK | pyiqa, pre-downloaded via hf-mirror |
| DOVER | OK | metrics/DOVER/, 229M weights |
| FastVQA | OK | metrics/FastVQA/, 122M weights + Swin-T 122M |


Fixes: downgraded setuptools<80, HF_ENDPOINT=hf-mirror.com, FastVQA symlink, cv2 video conversion

---

## 2. Datasets

| Dataset | Videos | Resolution | GT |
|---|---|---|---|
| UDM10 | 10 | 1272x720 | Yes |
| SPMCS | 30 | 960x536 | Yes |

---

## 3. Results (no_ref mode)

### Per-Dataset

| Metric | Type | UDM10 | SPMCS | 3×Smoke test videos |
|---|---|---|---|---|
| PSNR | FR | 29.66 | 18.99 | — |
| SSIM | FR | 0.868 | 0.490 | — |
| LPIPS | FR | 0.147 | 0.220 | — |
| DISTS | FR | 0.100 | 0.141 | — |
| CLIPIQA | NR | 0.454 | 0.545 | 0.519 |
| MUSIQ | NR | 59.57 | 67.57 | 59.71 |
| DOVER Technical | NR | 0.102 | 0.079 | 0.110 |
| DOVER Aesthetic | NR | 0.988 | 0.968 | 0.970 |
| DOVER Overall | NR | 0.618 | 0.498 | 0.590 |
| FastVQA | NR | 0.805 | 0.703 | 0.835 |

FR=Full-Reference (needs GT), NR=No-Reference (blind)
LPIPS/DISTS: lower is better. All others: higher is better.

### Key Findings
- UDM10 scores much higher on FR metrics (PSNR 29.66 vs 18.99) — easier dataset
- DOVER Aesthetic is near-perfect on all (0.97-0.99) — content quality preserved
- DOVER Technical is very low on all (0.08-0.11) — no_ref mode has technical artifacts
- No-ref mode is a baseline; paper recommends api/pisasr modes for best quality

---

## 4. PiSA-SR Mode Results (pisasr, ref_indices=0)

### 4.1 PiSA-SR Environment

| Component | Status |
|---|---|
| Conda env | PiSA-SR (Python 3.10, torch 2.0.1) |
| SD-2.1-base | preset/models/stable-diffusion-2-1-base |
| pisa_sr.pkl | 32M |
| RAM model | 5.3G |
| Fixes | numpy 1.23.5, huggingface_hub 0.25.2, loralib, fairscale |

### 4.2 UDM10 pisasr (10 videos)

| Metric | Type | Score |
|---|---|---|
| PSNR | FR | 28.72 dB |
| SSIM | FR | 0.841 |
| LPIPS | FR | 0.194 |
| DISTS | FR | 0.130 |
| CLIPIQA | NR | 0.294 |
| MUSIQ | NR | 50.86 |
| DOVER Technical | NR | 0.069 |
| DOVER Aesthetic | NR | 0.983 |
| DOVER Overall | NR | 0.511 |
| FastVQA | NR | 0.696 |

### 4.3 SPMCS pisasr (30 videos)

| Metric | Type | Score |
|---|---|---|
| PSNR | FR | 17.12 dB |
| SSIM | FR | 0.410 |
| LPIPS | FR | 0.292 |
| DISTS | FR | 0.185 |
| CLIPIQA | NR | 0.706 |
| MUSIQ | NR | 73.92 |
| DOVER Technical | NR | 0.081 |
| DOVER Aesthetic | NR | 0.976 |
| DOVER Overall | NR | 0.548 |
| FastVQA | NR | 0.722 |

### 4.4 no_ref vs pisasr Comparison

| Metric | UDM10 no_ref | UDM10 pisasr | SPMCS no_ref | SPMCS pisasr |
|---|---|---|---|---|
| PSNR | 29.66 | 28.72 | 18.99 | 17.12 |
| SSIM | 0.868 | 0.841 | 0.490 | 0.410 |
| LPIPS | 0.147 | 0.194 | 0.220 | 0.292 |
| DISTS | 0.100 | 0.130 | 0.141 | 0.185 |
| CLIPIQA | 0.454 | 0.294 | 0.545 | 0.706 |
| MUSIQ | 59.57 | 50.86 | 67.57 | 73.92 |
| DOVER Technical | 0.102 | 0.069 | 0.079 | 0.081 |
| DOVER Aesthetic | 0.988 | 0.983 | 0.968 | 0.976 |
| DOVER Overall | 0.618 | 0.511 | 0.498 | 0.548 |
| FastVQA | 0.805 | 0.696 | 0.703 | 0.722 |

### 4.5 Key Findings
- **FR metrics drop with pisasr**: PSNR/SSIM lower because PiSA-SR generates perceptually-enhanced keyframes that differ pixel-wise from GT
- **NR metrics improve on harder datasets**: SPMCS CLIPIQA +0.161, MUSIQ +6.35 with pisasr
- **NR metrics drop on easy datasets**: UDM10 CLIPIQA -0.160 — PiSA-SR may over-enhance already-clean synthetic data
- **DOVER Aesthetic stays 0.97+ across all modes**: visual appeal is consistently high
- **PiSA-SR benefits real-world-like degradation (SPMCS) over clean synthetic (UDM10)**
