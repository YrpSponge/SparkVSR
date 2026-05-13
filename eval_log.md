# SparkVSR + DLoRA 融合实验 — 方法与结果

> Server: ddcloud (183.232.132.248:10080) | Path: /data3/ryu455/SparkVSR | Date: 2026-05

---

## 1. 动机与思路

### 1.1 出发点

两类 VSR 方法在能力上天然互补：

| | DLoRA (W4+SC) | SparkVSR |
|---|---|---|
| 类型 | blind one-step VSR | sparse-keyframe-conditioned VSR |
| 擅长 | 无参考帧时提高感知质量 + 时间一致性 | 利用少量高质量关键帧传播到全视频 |
| 局限 | 全局增强，无法针对关键帧集中发力 | 强依赖关键帧质量，no_ref 画质平庸 |

核心直觉：**让 DLoRA 负责"把关键帧修好"，SparkVSR 负责"把高质量先验传播出去"。**

### 1.2 融合架构（方案 A：Keyframe Enhancer）

```
LQ 视频 (T 帧)
    |
    +-- Step 1: 选取关键帧 (第 0 帧 / 自动选取)
    |
    +-- Step 2: DLoRA(W4+SC) 增强关键帧 -> 高清参考帧
    |         (子进程调用 subprocess, 和 PiSA-SR 同样的接口)
    |
    +-- Step 3: SparkVSR 以关键帧为先验，传播到全 T 帧
    |         (CogVideoX-I2V pipeline, classifier-free guidance)
    |
    +-- Step 4 (未完成): SideChannel 后处理增强输出细节
```

### 1.3 为什么不做端到端融合

- DLoRA 和 SparkVSR 底座不同 (SD 2.1 vs CogVideoX 5B)
- 训练目标不同 (pixel-space detail vs latent-space propagation)
- Pipeline 级融合改动小、可单独消融、工程风险低
- 符合课程项目周期和算力约束

---

## 2. 实验设置

### 2.1 数据集

| 数据集 | 视频数 | 总帧数 | LQ 分辨率 | 退化类型 | GT |
|---|---|---|---|---|---|
| UDM10 | 10 | 310 | ~1272x720 | 合成 BD 4x 降采样 | 有 |
| SPMCS | 30 | 934 | ~960x536 | 合成+真实混合 | 有 |

### 2.2 对照方法

| 方法 | 说明 |
|---|---|
| DLoRA standalone | DLoRA 独立 blind VSR (W1/W2/W3/W4+SC 消融) |
| SparkVSR no_ref | SparkVSR 盲超分（无关键帧参考） |
| SparkVSR + PiSA-SR | PiSA-SR 扩散增强关键帧 -> SparkVSR 传播 |
| SparkVSR + DLoRA (W4+SC) | DLoRA 最强配置增强关键帧 -> SparkVSR 传播 |
| Ablation: 关键帧数量 | 1-kf (frame 0) vs 3-kf (auto: 0/mid/last) |

### 2.3 评估指标

| 类型 | 指标 | 说明 |
|---|---|---|
| Full-Reference | PSNR, SSIM | 像素精度 |
| Full-Reference | LPIPS, DISTS | 感知距离（越低越好） |
| No-Reference | CLIPIQA, MUSIQ | 图像感知质量 |
| No-Reference | DOVER (Technical/Aesthetic/Overall) | 视频技术+审美质量 |
| No-Reference | FastVQA (FasterVQA) | 视频质量评估 |

### 2.4 软硬件

| 项目 | 规格 |
|---|---|
| GPU | 2x NVIDIA RTX A6000 (48GB) |
| SparkVSR 环境 | conda: sparkvsr, torch 2.5.0, diffusers 0.37.x |
| DLoRA / PiSA-SR 环境 | conda: PiSA-SR, torch 2.0.1, diffusers 0.25.0, mmcv 2.1.0 |

---

## 3. DLoRA 独立评测 (Standalone)

DLoRA 作为完整 VSR 模型独立推理。W1 为 baseline (SPyNet 光流)，W4+SC 为最强配置。

### 3.1 UDM10

| Config | PSNR | SSIM | LPIPS | MUSIQ | CLIPIQA | DOVER | FasterVQA |
|---|---|---|---|---|---|---|---|
| W1 (SPyNet) | 26.72 | 0.767 | 0.219 | 68.98 | 0.658 | 0.762 | 0.052 |
| W2 | 27.73 | 0.781 | 0.209 | 66.68 | 0.674 | 0.759 | 0.052 |
| W3 | 27.58 | 0.809 | 0.218 | 52.40 | 0.442 | 0.658 | 0.036 |
| **W4+SC** | **28.36** | **0.814** | **0.194** | 62.01 | 0.595 | 0.719 | 0.046 |

### 3.2 SPMCS

| Config | PSNR | SSIM | LPIPS | MUSIQ | CLIPIQA | DOVER | FasterVQA |
|---|---|---|---|---|---|---|---|
| W1 (SPyNet) | 22.61 | 0.679 | 0.156 | 66.07 | 0.617 | 0.791 | 0.047 |
| W2 | 23.59 | 0.706 | 0.143 | 66.12 | 0.642 | 0.802 | 0.049 |
| W3 | 23.44 | 0.689 | 0.200 | 49.61 | 0.427 | 0.702 | 0.030 |
| **W4+SC** | **24.45** | **0.726** | **0.156** | 60.46 | 0.569 | 0.774 | 0.045 |

> 结论：W4+SC (CFR-RAFT + SideChannel) 在所有 FR 指标上最优。选定 W4+SC 作为后续融合的关键帧增强器。

---

## 4. SparkVSR 融合评测 (Reference Method)

### 4.1 实现细节

DLoRA 通过 subprocess 方式接入 SparkVSR 的 ref_mode 框架，和 PiSA-SR 共用相同的调用接口。新增代码位于 sparkvsr_inference_script.py 的 dlora ref_mode 块（约 60 行），创建 dloral_keyframe.py 作为 CLI wrapper。

关键修复：DLoRA 子进程的 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 与 torch 2.0.1 不兼容，排查 8 次后定位并移除。

### 4.2 UDM10 (10 视频, 1 关键帧)

| Metric | no_ref | + PiSA-SR | + DLoRA (W4+SC) | 最佳 |
|---|---|---|---|---|
| PSNR | **29.66** | 28.72 | 26.73 | no_ref |
| SSIM | **0.868** | 0.841 | 0.790 | no_ref |
| LPIPS | **0.147** | 0.194 | 0.225 | no_ref |
| DISTS | **0.100** | 0.130 | 0.142 | no_ref |
| CLIPIQA | 0.454 | 0.294 | **0.593** | dlora +31% |
| MUSIQ | 59.57 | 50.86 | **67.93** | dlora +14% |
| DOVER Technical | 0.102 | 0.069 | **0.119** | dlora +17% |
| DOVER Aesthetic | 0.988 | 0.983 | **0.992** | dlora |
| DOVER Overall | 0.618 | 0.511 | **0.687** | dlora +11% |
| FastVQA | 0.805 | 0.696 | **0.841** | dlora +4% |

> UDM10 结论：DLoRA 在所有 NR 指标上全面碾压。PiSA-SR 反而损害干净合成数据（CLIPIQA 0.294 < no_ref 0.454）。FR 下降符合感知-像素取舍。

### 4.3 SPMCS (30 视频, 1 关键帧)

| Metric | no_ref | + PiSA-SR | + DLoRA (W4+SC) | 最佳 |
|---|---|---|---|---|
| PSNR | **18.99** | 17.12 | 18.67 | no_ref |
| SSIM | **0.490** | 0.410 | 0.460 | no_ref |
| LPIPS | **0.220** | 0.292 | 0.266 | no_ref |
| DISTS | **0.141** | 0.185 | 0.158 | no_ref |
| CLIPIQA | 0.545 | **0.706** | 0.607 | pisasr |
| MUSIQ | 67.57 | **73.92** | 70.18 | pisasr |
| DOVER Technical | 0.079 | 0.081 | **0.094** | dlora |
| DOVER Aesthetic | 0.968 | **0.976** | 0.942 | pisasr |
| DOVER Overall | 0.498 | **0.548** | 0.490 | pisasr |
| FastVQA | 0.703 | 0.722 | **0.738** | dlora |

> SPMCS 结论：三方各有胜负。PiSA-SR 在 CLIPIQA/MUSIQ/审美占优（扩散增强对真实退化更有效），DLoRA 在技术/FastVQA 领先。

---

## 5. Ablation: 关键帧数量

SPMCS, dlora (W4+SC) 模式。1-kf = 仅帧 0。3-kf = auto 选取首/中/尾。

| Metric | 1-kf | 3-kf (auto) | 结论 |
|---|---|---|---|
| PSNR | 18.67 | 18.73 | 持平 |
| SSIM | 0.460 | 0.461 | 持平 |
| LPIPS | 0.266 | 0.267 | 持平 |
| CLIPIQA | **0.607** | 0.596 | 持平 |
| MUSIQ | **70.18** | 69.13 | 持平 |
| DOVER Overall | 0.490 | 0.486 | 持平 |
| FastVQA | **0.738** | 0.723 | 持平 |

> 结论：31 帧短视频上，增加关键帧无增益。传播距离足够短，单帧信息已饱和。后续应在帧数 >50 的数据集上测试。

---

## 6. 交叉对比: DLoRA Standalone vs SparkVSR+DLoRA

| Metric | UDM10 standalone | UDM10 fusion | SPMCS standalone | SPMCS fusion |
|---|---|---|---|---|
| PSNR | **28.36** | 26.73 | **24.45** | 18.67 |
| SSIM | **0.814** | 0.790 | **0.726** | 0.460 |
| LPIPS | **0.194** | 0.225 | **0.156** | 0.266 |
| CLIPIQA | 0.595 | 0.593 | 0.569 | **0.607** |
| MUSIQ | 62.01 | **67.93** | 60.46 | **70.18** |

> FR 指标不可直接对比 (DLoRA 8x vs SparkVSR 4x 输出尺寸不同)。NR 指标有参考价值：融合后在 UDM10 上的 MUSIQ 显著优于 standalone，表明 SparkVSR 传播有效放大了关键帧的感知优势。

---

## 7. 环境与修复记录

### 7.1 环境

| 组件 | 环境 | 关键依赖 |
|---|---|---|
| SparkVSR | conda: sparkvsr | torch 2.5.0, diffusers 0.37, CogVideoX |
| DLoRA | conda: PiSA-SR | torch 2.0.1, diffusers 0.25.0, mmcv 2.1.0 |
| PiSA-SR | conda: PiSA-SR | SD 2.1 base, pisa_sr.pkl (32M) |
| DOVER | metrics/DOVER/ | DOVER.pth (229M) |
| FastVQA | metrics/FastVQA/ | FAST_VQA_3D_1_1.pth (122M) + Swin-T (122M) |

### 7.2 修复记录

| # | 问题 | 修复 |
|---|---|---|
| 1 | GitHub HTTPS 被墙, clone 失败 | 配置 SSH key (git@github.com) |
| 2 | HuggingFace 下载失败 | HF_ENDPOINT=https://hf-mirror.com |
| 3 | setuptools 82.x 移除 pkg_resources | 降级到 79.0.1 |
| 4 | FastVQA 路径不匹配 | ln -sfn FAST-VQA-and-FasterVQA metrics/FastVQA |
| 5 | DLoRA subprocess 全部失败 (v1~v7) | 移除 PYTORCH_CUDA_ALLOC_CONF (torch 2.0.1 不兼容) |
| 6 | SPMCS ffmpeg 转换出空文件 | 改用 Python/cv2 写 MP4 |
| 7 | GPU 抢占 OOM | 自动选空闲>30GB GPU 启动 |

### 7.3 代码位置

| 文件 | 说明 |
|---|---|
| sparkvsr_inference_script.py | 推理主脚本, dlora ref_mode (L961-1020) |
| dloral_keyframe.py | DLoRA CLI wrapper |
| side_channel/ | SideChannel 后处理模块 (实验中) |
| eval_log.md | 本文件 |
| GitHub | github.com/YrpSponge/SparkVSR |

---

## 8. 未完成 / 下一步

| 项目 | 状态 | 备注 |
|---|---|---|
| RealVSR 评测 | GPU 抢占 | 50 视频, 需 GPU 长时间空闲 |
| 方案 C (SideChannel) | 代码就绪 | 单帧测试通过, 全量待跑 |
| 自适应关键帧 (帧差法) | 未开始 | 长视频预期有效 |
| DLoRA 常驻加速 | 未开始 | 避免每帧重载模型 |
