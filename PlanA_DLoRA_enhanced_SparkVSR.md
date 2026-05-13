# Plan A：DLoRA-enhanced SparkVSR 方案梳理

## 1. 项目背景

我们当前的核心目标不是单纯提出一个“概念上很新”的模型，而是：

- 在课程项目周期内做出**可运行、可比较、可复现**的系统；
- 尽可能在现有 benchmark 上取得**更好的指标表现**；
- 最终结果不仅要有数值提升，也要更接近我们想要的“高质量视频修复 / 4K 修复感”。

在前期调研中，我们重点关注了两类方法：

1. **blind / no-reference 的 one-step VSR 路线**，代表方法是 **DLoRA**；
2. **reference-guided / sparse-keyframe-conditioned 的 VSR 路线**，代表方法是 **SparkVSR**。

经过分析，我们发现这两类方法并不是互斥关系，反而在功能上天然互补：

- **DLoRA** 擅长在没有外部参考帧的情况下，提高视频的**感知质量（perceptual quality）**，同时兼顾**时间一致性（temporal consistency）**；
- **SparkVSR** 擅长利用少量高质量关键帧（keyframes）作为先验，把这种高质量视觉特征传播到整段视频中。

因此，我们计划优先尝试一个融合思路，即：

> **方案 A：DLoRA-enhanced SparkVSR**
>
> 使用改进后的 DLoRA 生成更高质量的关键帧参考（references），再由 SparkVSR 将这些高质量先验传播到整段视频中。

---

## 2. 两个基础方法分别在做什么

### 2.1 DLoRA 的作用

DLoRA 是一条 **blind one-step Real-VSR** 路线。其核心思想是：

- 将视频超分中的两个关键目标显式拆开：
  - **temporal consistency**（前后帧稳定、不闪烁）
  - **detail enhancement**（高频纹理更丰富、画面更真实）
- 分别用不同的 LoRA 分支来学习这两个目标，再在推理时合并。

从功能角度看，DLoRA 更像是一个：

> **强 blind perceptual enhancer**

也就是说，即使没有参考帧，它也能尽量把一段低质量视频修得“更像真的”。

### 2.2 SparkVSR 的作用

SparkVSR 不是纯 blind VSR，而是一个 **sparse-keyframe-conditioned video super-resolution** 框架。其核心逻辑是：

1. 从视频中选取少量关键帧；
2. 先用任意现成 ISR 模型把这些关键帧修好；
3. 再利用这些关键帧作为视觉锚点，把高质量先验传播到整段视频。

从功能角度看，SparkVSR 更像是一个：

> **高质量 keyframe prior 的传播器（propagator）**

它真正擅长的不是“凭空修好每一帧”，而是“把少量高质量参考帧中的好细节稳定地传到整段视频中”。

---

## 3. 方案 A 的核心思想

方案 A 的基本思路是：

> **不直接改 SparkVSR 的大 backbone，先替换它的 reference source。**

也就是说，我们让改进后的 DLoRA 不再只是一个独立的 blind VSR 模型，而是让它在整个系统里承担一个新的角色：

> **作为 SparkVSR 的 keyframe/reference generator**

### 3.1 直观理解

我们把整个系统分成两个阶段：

#### 阶段一：生成高质量关键帧

- 从输入 LR video 中选出少量 sparse keyframes；
- 使用改进后的 DLoRA 对这些关键帧进行增强；
- 得到更高质量、更高感知表现的参考帧。

#### 阶段二：传播高质量先验

- 将这些 DLoRA 生成的高质量关键帧输入给 SparkVSR；
- 由 SparkVSR 将关键帧中的高质量纹理、结构和视觉风格传播到整段视频；
- 最终得到整段增强后的视频结果。

### 3.2 一句话概括

> **DLoRA 负责“造好老师”，SparkVSR 负责“让老师带全班”。**

---

## 4. 为什么这个方案合理

### 4.1 两个方法的能力互补

DLoRA 和 SparkVSR 的强项正好位于系统的不同层级：

- DLoRA：强在**关键帧本身修得好不好**；
- SparkVSR：强在**这些高质量关键帧能否传播到整段视频**。

因此，这个融合不是硬拼两个大模型，而是非常自然的模块分工。

### 4.2 不需要立即做高风险端到端融合

如果直接把 DLoRA 的内部模块和 SparkVSR 的 backbone 从底层硬融合，会遇到：

- 底座不统一；
- 训练目标不统一；
- 算力和工程复杂度过高。

而方案 A 只是在 pipeline 级别做融合：

- SparkVSR 主体保留；
- DLoRA 作为前端参考生成器接入；
- 工程风险低很多，更适合课程项目。

### 4.3 更容易做出指标提升

SparkVSR 的感知/VQA 指标表现本来就很强，而其结果质量高度依赖 reference frame quality。

因此，如果我们能把 reference frame 从普通 ISR 提升为“改进后的 DLoRA 输出”，理论上最有希望进一步提升：

- **MUSIQ**
- **CLIP-IQA**
- **DOVER / FasterVQA**

也就是说，这个方案在“课程项目追求指标表现”的目标下，性价比很高。

---

## 5. 方案 A 与方案 B 的区别

为了防止后续讨论混淆，我们这里顺带明确一下方案 A 和方案 B 的区别。

### 方案 A

- 先选定关键帧；
- 再用 DLoRA 去增强这些关键帧；
- SparkVSR 负责传播。

重点是：

> **提升已有 keyframe 的质量**

### 方案 B

- 没有现成外部 reference 时；
- 先由 DLoRA 从 LR video 自动生成 pseudo references；
- 再让 SparkVSR 传播。

重点是：

> **让系统自动生成 reference**

### 当前选择理由

我们当前优先做 **方案 A**，因为：

- 更容易落地；
- 改动更局部；
- 更适合快速做 ablation；
- 更容易在当前资源下拿到可解释的指标提升。

---

## 6. 方法流程（当前版本）

下面是我们当前计划采用的流程：

```text
Input: LR video
    |
    |-- Step 1: Keyframe Selection
    |      选出 sparse keyframes（手动 / I-frame / quality-aware 等）
    |
    |-- Step 2: Keyframe Enhancement by Improved DLoRA
    |      使用改进后的 DLoRA 对关键帧做增强
    |      输出 high-quality reference frames
    |
    |-- Step 3: SparkVSR Propagation
    |      将参考帧输入 SparkVSR
    |      将高质量先验传播到整段视频
    |
    |-- Step 4: Guidance Scheduling (optional)
    |      调整 SparkVSR 中 reference guidance 的强度
    |
Output: Enhanced HR video
```

---

## 7. 我们当前最值得做的改进点

围绕方案 A，我们目前认为有三类最值得优先尝试的改进。

### 7.1 改进点一：更强的 DLoRA 关键帧增强器

这是当前最直接的主线。

目标是让输入 SparkVSR 的 keyframe 参考帧质量尽可能高。

我们可以利用同伴当前正在做的 DLoRA 模块改进，例如：

- 更强的 detail branch；
- 更好的 temporal/detail 分工；
- 更适合真实退化的结构设计；
- 更强的感知损失或高频损失。

这部分的作用是：

> **提高 SparkVSR 的 reference source 上限**

### 7.2 改进点二：更好的 keyframe selection

即使 reference generator 很强，如果选错关键帧，传播效果也会受限。

因此可以进一步探索：

- random vs I-frame vs manual；
- quality-aware keyframe selection；
- motion-aware keyframe selection；
- 不同视频长度对应的 keyframe 数量策略。

这部分的作用是：

> **挑出最适合当“高质量锚点”的帧**

### 7.3 改进点三：reference guidance 调度

SparkVSR 本身带有 `reference-free guidance` 机制，可以控制输出对 reference 的依赖强度。

这里可以尝试：

- 固定 guidance scale；
- 分段 guidance scale；
- 按视频内容自适应 guidance scale；
- 比较不同 guidance scale 对 PSNR / perceptual quality 的影响。

这部分的作用是：

> **平衡“更贴近 reference”与“保留 blind restoration 自由度”**

---

## 8. 实验设计建议

### 8.1 Baseline 设定

建议至少准备以下对照：

1. **SparkVSR + 原始 reference source**（比如 PiSA-SR 或默认方案）
2. **SparkVSR + 改进后的 DLoRA reference**
3. （可选）**SparkVSR no-ref**
4. （可选）**DLoRA standalone**

这样可以清楚回答两个问题：

- 改进后的 DLoRA 作为 reference source 是否真的更强？
- SparkVSR 传播是否真正放大了这种提升？

### 8.2 建议关注的指标

根据 SparkVSR 路线的特点，建议重点关注：

#### 感知 / 视频质量指标
- **MUSIQ**
- **CLIP-IQA**
- **DOVER**
- **FasterVQA**

#### 如果有 GT 的数据集
- **PSNR**
- **SSIM**
- **LPIPS**

我们预期最可能明显上涨的是：

- 感知相关指标（MUSIQ、CLIP-IQA）
- 视频质量相关指标（DOVER、FasterVQA）

### 8.3 Ablation 建议

建议逐步做 ablation：

- Baseline SparkVSR
- + DLoRA reference
- + improved DLoRA reference
- + smarter keyframe selection
- + adaptive guidance scheduling

这样能够清楚展示：每一个模块改动分别带来了多少收益。

---

## 9. 预期优势

如果方案 A 成功，我们预期可以得到以下优势：

### 9.1 在指标上

有机会继续提升 SparkVSR 已经较强的：

- CLIP-IQA
- MUSIQ
- DOVER
- FasterVQA

### 9.2 在方法上

我们的方法故事会非常清晰：

> 我们将强 blind perceptual prior（改进后的 DLoRA）与 sparse-keyframe-conditioned propagation（SparkVSR）结合，构建了一个 hybrid VSR pipeline。

### 9.3 在工程上

相比完全端到端重新设计 backbone：

- 更适合当前硬件条件；
- 更容易跑通；
- 更适合课程项目周期；
- 更容易做对比和消融实验。

---

## 10. 当前风险与注意事项

### 10.1 输入设定不是完全 blind

一旦使用 reference-guided SparkVSR，最终 claim 就不能写成“统一公平地超越所有 blind SOTA”。

更准确的说法应该是：

- 我们在 **reference-guided / sparse-keyframe-conditioned** 设定下取得了提升；
- 或者我们提出了一个 **blind prior + keyframe propagation** 的 hybrid framework。

### 10.2 reference 质量和传播可能会冲突

如果 DLoRA 生成的 keyframe 中有伪影，SparkVSR 可能会把错误传播到整段视频。

因此，guidance scale 和 keyframe 质量控制会非常重要。

### 10.3 评价必须分清主表和补充表

如果后续做实验汇报，建议将：

- blind/no-ref 对比
- reference-guided 对比

分开成不同的表或不同的实验设置，避免不同输入设定混在一起造成不公平比较。

---

## 11. 下一步执行计划

### 阶段 1：先验证可行性

- 跑通 SparkVSR baseline
- 跑通同伴当前改进版 DLoRA
- 用 DLoRA 生成 sparse keyframe references
- 接入 SparkVSR 做初步融合推理

### 阶段 2：做 reference source 对比

比较：

- baseline reference source
- 改进版 DLoRA reference

重点看是否能提升 MUSIQ / CLIP-IQA / DOVER / FasterVQA。

### 阶段 3：做轻量改进（future work）

- keyframe selection
- guidance scale 调度
- reference 数量和分布策略



