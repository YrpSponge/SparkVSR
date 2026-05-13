# SparkVSR + DLoRA 融合实验 — 完整记录

> Server: ddcloud | Path: /data3/ryu455/SparkVSR | Date: 2026-05

---

## 1. 数据集

| 数据集 | 视频数 | 总帧数 | 输入分辨率 | 退化类型 | GT |
|---|---|---|---|---|---|
| UDM10 | 10 | 310 | ~1272×720 | 合成 BD 降采样 | 有 |
| SPMCS | 30 | 934 | ~960×536 | 合成 + 真实混合 | 有 |

---

## 2. DLoRA 独立评测 (Standalone)

DLoRA 作为 blind one-step VSR 模型独立推理。W1 为 baseline（SPyNet 光流），W4+SC 为最强配置（CFR-RAFT + SideChannel）。

### 2.1 UDM10 — Full-Reference

| Config | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Δ PSNR vs W1 |
|---|---|---|---|---|
| W1 (SPyNet) | 26.718 | 0.767 | 0.219 | — |
| W2 | 27.726 | 0.781 | 0.209 | +1.01 |
| W3 | 27.579 | 0.809 | 0.218 | +0.86 |
| **W4+SC** | **28.358** | **0.814** | **0.194** | **+1.64** |

### 2.2 UDM10 — No-Reference

| Config | MUSIQ ↑ | CLIPIQA ↑ | DOVER ↑ | FasterVQA ↑ |
|---|---|---|---|---|
| W1 | 68.98 | 0.658 | 0.762 | 0.052 |
| W2 | 66.68 | 0.674 | 0.759 | 0.052 |
| W3 | 52.40 | 0.442 | 0.658 | 0.036 |
| **W4+SC** | **62.01** | **0.595** | **0.719** | **0.046** |

### 2.3 SPMCS — Full-Reference

| Config | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Δ PSNR vs W1 |
|---|---|---|---|---|
| W1 (SPyNet) | 22.612 | 0.679 | 0.156 | — |
| W2 | 23.592 | 0.706 | 0.143 | +0.98 |
| W3 | 23.442 | 0.689 | 0.200 | +0.83 |
| **W4+SC** | **24.450** | **0.726** | **0.156** | **+1.84** |

### 2.4 SPMCS — No-Reference

| Config | MUSIQ ↑ | CLIPIQA ↑ | DOVER ↑ | FasterVQA ↑ |
|---|---|---|---|---|
| W1 | 66.07 | 0.617 | 0.791 | 0.047 |
| W2 | 66.12 | 0.642 | 0.802 | 0.049 |
| W3 | 49.61 | 0.427 | 0.702 | 0.030 |
| **W4+SC** | **60.46** | **0.569** | **0.774** | **0.045** |

> W4+SC = CFR-RAFT optical flow + Detail SideChannel. 在所有 FR 指标上最优，NR 指标 W2 略优。

---

## 3. SparkVSR 融合评测 (Reference Method)

三种 ref_mode 对比：no_ref（盲超分）、pisasr（PiSA-SR 增强关键帧）、dlora（DLoRA W4+SC 增强关键帧，1 关键帧）。

### 3.1 UDM10 — Full-Reference

| Metric | no_ref | pisasr | dlora (W4+SC) |
|---|---|---|---|
| PSNR ↑ | **29.66** | 28.72 | 26.73 |
| SSIM ↑ | **0.868** | 0.841 | 0.790 |
| LPIPS ↓ | **0.147** | 0.194 | 0.225 |
| DISTS ↓ | **0.100** | 0.130 | 0.142 |

### 3.2 UDM10 — No-Reference

| Metric | no_ref | pisasr | dlora (W4+SC) |
|---|---|---|---|
| CLIPIQA ↑ | 0.454 | 0.294 | **0.593** |
| MUSIQ ↑ | 59.57 | 50.86 | **67.93** |
| DOVER Technical ↑ | 0.102 | 0.069 | **0.119** |
| DOVER Aesthetic ↑ | 0.988 | 0.983 | **0.992** |
| DOVER Overall ↑ | 0.618 | 0.511 | **0.687** |
| FastVQA ↑ | 0.805 | 0.696 | **0.841** |

### 3.3 SPMCS — Full-Reference

| Metric | no_ref | pisasr | dlora (W4+SC) |
|---|---|---|---|
| PSNR ↑ | **18.99** | 17.12 | 18.67 |
| SSIM ↑ | **0.490** | 0.410 | 0.460 |
| LPIPS ↓ | **0.220** | 0.292 | 0.266 |
| DISTS ↓ | **0.141** | 0.185 | 0.158 |

### 3.4 SPMCS — No-Reference

| Metric | no_ref | pisasr | dlora (W4+SC) |
|---|---|---|---|
| CLIPIQA ↑ | 0.545 | **0.706** | 0.607 |
| MUSIQ ↑ | 67.57 | **73.92** | 70.18 |
| DOVER Technical ↑ | 0.079 | 0.081 | **0.094** |
| DOVER Aesthetic ↑ | 0.968 | **0.976** | 0.942 |
| DOVER Overall ↑ | 0.498 | **0.548** | 0.490 |
| FastVQA ↑ | 0.703 | 0.722 | **0.738** |

---

## 4. Ablation: 关键帧数量 (SPMCS, dlora 模式)

| Metric | 1-kf (frame 0) | 3-kf (auto: 0/mid/last) | Delta |
|---|---|---|---|
| PSNR ↑ | 18.67 | **18.73** | +0.06 |
| SSIM ↑ | 0.460 | **0.461** | +0.001 |
| LPIPS ↓ | **0.266** | 0.267 | +0.001 |
| DISTS ↓ | **0.158** | 0.164 | +0.006 |
| CLIPIQA ↑ | **0.607** | 0.596 | -0.011 |
| MUSIQ ↑ | **70.18** | 69.13 | -1.05 |
| DOVER Technical ↑ | **0.094** | 0.089 | -0.005 |
| DOVER Aesthetic ↑ | 0.942 | **0.943** | +0.001 |
| DOVER Overall ↑ | **0.490** | 0.486 | -0.004 |
| FastVQA ↑ | **0.738** | 0.723 | -0.015 |

> 结论：31 帧短视频上 3 关键帧无增益。传播距离足够短，单个关键帧已饱和。

---

## 5. DLoRA Standalone vs SparkVSR+DLoRA 交叉对比

| Metric | UDM10 DLoRA standalone | UDM10 SparkVSR+DLoRA | SPMCS DLoRA standalone | SPMCS SparkVSR+DLoRA |
|---|---|---|---|---|
| PSNR ↑ | **28.36** | 26.73 | **24.45** | 18.67 |
| SSIM ↑ | **0.814** | 0.790 | **0.726** | 0.460 |
| LPIPS ↓ | **0.194** | 0.225 | **0.156** | 0.266 |
| CLIPIQA ↑ | 0.595 | **0.593** | 0.569 | **0.607** |
| MUSIQ ↑ | 62.01 | **67.93** | 60.46 | **70.18** |
| DOVER ↑ | 0.719 | **0.687** | **0.774** | 0.490 |
| FastVQA ↑ | 0.046 | **0.841** | 0.045 | **0.738** |

> 注意：DLoRA standalone 和 SparkVSR+DLoRA 的 FR 指标不可直接对比——DLoRA 输出 8× 超分经缩放匹配 GT，SparkVSR 输出 4×。NR 指标可横向参考。FastVQA 数值差异来自不同版本的指标实现。

---

## 6. 环境搭建

| 组件 | 环境 | 关键依赖 |
|---|---|---|
| SparkVSR | conda: sparkvsr | torch 2.5.0, diffusers 0.37.x |
| DLoRA / PiSA-SR | conda: PiSA-SR | torch 2.0.1, diffusers 0.25.0, mmcv 2.1.0 |
| DOVER | metrics/DOVER/ | pretrained_weights/DOVER.pth (229M) |
| FastVQA | metrics/FastVQA/ | pretrained_weights/FAST_VQA_3D_1_1.pth (122M) + Swin-T (122M) |

### 修复记录

| 问题 | 修复 |
|---|---|
| GitHub HTTPS 被墙 | 改用 SSH (git@github.com) |
| HuggingFace 被墙 |  |
| setuptools 82.x 移除 pkg_resources | 降级到 79.0.1 |
| FastVQA 路径不匹配 |  |
| DLoRA subprocess PYTORCH_CUDA_ALLOC_CONF | 移除 (torch 2.0.1 不兼容 expandable_segments) |
| SPMCS ffmpeg 转换失败 | 改用 Python/cv2 写 MP4 |

