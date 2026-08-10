# 球场感知双 Head 模型竞赛

目标不是在 Cityscapes 排名中选网络，而是在同一足球数据、同一输入尺寸和同一标注协议下，同时比较细线/关键点定位、遮挡鲁棒性和 RTX 5060 Ti 延迟。

## 固定输出契约

所有候选模型必须输出：

```text
semantic_logits: [N, 21, H, W]
landmark_heatmaps: [N, 33, H/4, W/4]
landmark_offsets: [N, 66, H/4, W/4]
```

- 20 个带身份的球场直线/圆弧类别，加 1 个背景类；
- 兼容现有关键点检测器的 32 个公制点，加中心点，共 33 个 landmark；
- offset 每点 2 通道，只在热图峰值位置监督；
- 模型输入和训练标签共用 `PitchModel`/`PitchPerceptionVocabulary`，禁止单独维护标签顺序。

## 候选

| 编号 | 候选 | 当前状态 | 用途 |
|---|---|---|---|
| C0 | 现有 32 点模型 | 已有权重和兼容适配 | 当前准确率/速度基线 |
| C1 | MobileNetV3-Large 双 head | 架构、数据、训练、推理、CUDA 基准已实现 | 最低成本候选 |
| C2 | PIDNet-S 双 head | 待同数据实现/训练 | 细线 detail 主候选 |
| C3 | SegFormer-B0 双 head | 待同数据实现/训练 | 全局上下文对照 |
| C4 | PnLCalib 官方模型 | 仅评测适配 | 精度上界/速度参考，不作生产默认 |

未实现候选在 benchmark 中必须显示 `not_implemented`，不得静默替换为 C1。

## 数据契约

`PitchRegistrationDataset`：

- 只加载指定 `train` 或 `validation` split；
- 同一源视频序列不能重复作为多个 manifest 输入；
- 先验证人工锚点，再拟合该帧真值变换；
- 从公制球场模型自动投影全部语义线/圆弧和 landmark；
- 语义 mask 在输入分辨率监督，landmark/offset 在 stride 4 监督；
- 只允许不会破坏几何关系的亮度、色彩、模糊、压缩和遮挡增强；
- 不允许未同步更新标注的随机透视或裁剪。

## C1 结构

```text
MobileNetV3-Large ImageNet backbone
  ├── stride-4 low feature (24 ch → 48 ch)
  └── stride-32 high feature (960 ch → 96 ch)
              ↓ bilinear upsample
       shared stride-4 decoder
          ├── semantic head
          ├── landmark heatmap head
          └── subpixel offset head
```

参数量：`3,253,384`。没有随机 128 维 embedding，也不引入与任务无关的分类头。

## 训练

```bash
uv run python -m experiments.field_registration.train_dual_head \
  --train-manifests /data/train-a/manifest.json /data/train-b/manifest.json \
  --validation-manifests /data/val-a/manifest.json \
  --output outputs/pitch-perception/mobilenet-v3-dual-head.pt \
  --device cuda \
  --batch-size 8 \
  --epochs 30
```

损失：

```text
L = semantic weighted cross-entropy
  + landmark modified focal loss
  + 0.25 × masked Smooth-L1 offset loss
```

训练脚本输出 validation foreground IoU 与 PCK@10px。最终报告还必须使用独立工具计算每类线距离、遮挡分组指标和注册误差，不能只根据训练 loss 选模型。

## CUDA 结构基准

环境：RTX 5060 Ti，输入 512×288，batch=1，FP16，30 次 warmup，150 次计时。

| 模型 | 权重 | mean | median | P95 | max | 峰值 allocated | 参数量 |
|---|---|---:|---:|---:|---:|---:|---:|
| C1 MobileNetV3 双 head | 随机，仅结构延迟 | 4.29 ms | 3.33 ms | 6.82 ms | 7.77 ms | 48.45 MB | 3.25 M |

该结果说明 C1 结构满足“单次 FP16 ≤15 ms、增量显存 ≤500 MB”的候选门槛。随机权重不具备任何准确率意义，不能写成“模型已有效识别球场”。

复现：

```bash
uv run python tools/benchmark_pitch_perception.py \
  --device cuda \
  --input-width 512 \
  --input-height 288 \
  --warmup 30 \
  --iterations 150 \
  --output pitch-perception-cuda.json
```

## 选择规则

生产候选必须同时满足：

1. 独立 test 序列 PCK@5/10px 与每类语义线距离；
2. 遮挡、阴影、边缘、快速 pan 分组不出现不可接受退化；
3. 下游 P95 线重投影和网格米制误差优于 C0；
4. 错误矩阵通过质量门禁率为 0；
5. FP16 单次 P95 ≤15 ms，完整管线 FPS 下降 ≤10%；
6. 30–60 分钟运行无持续显存增长。

在 C1/C2/C3 使用真实标注完成同数据训练前，不冻结生产模型。
