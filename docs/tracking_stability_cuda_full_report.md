# CUDA tracking stability benchmark

日期：2026-08-07

环境：远端 `AstraForge`，NVIDIA GeForce RTX 5060 Ti 16 GB，CUDA 可用，`torch 2.13.0+cu130`，依赖通过 `uv sync --frozen` 安装。

输入：`test1.mp4` 和 `test2.mp4`，`imgsz=640`。由于视频解码器实际产生的帧数与容器元数据不同，报告使用实际处理帧数。没有人工逐帧 ID ground truth，因此这里报告的是生产链路生命周期指标，不声称 IDF1/HOTA。

## Profile

| Profile | 检测 conf | NMS IoU | lost buffer | 短时预测 | 实体重连 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 current | 0.25 | 0.70 | 45 | 0 帧 | 关闭 |
| E1 tuned | 0.20 | 0.80 | 60 | 0 帧 | 关闭 |
| E2 prediction + entity | 0.20 | 0.80 | 60 | 6 帧 | 12 帧窗口 |

## Results

| 视频 | 实际帧数 | Profile | FPS | Track interruptions | Raw fragmentations | Entity fragmentations | ID switches | Predicted frames |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test1 | 1930 | E0 | 56.06 | 469 | 469 | 469 | 0 | 0 |
| test1 | 1930 | E1 | 65.10 | 602 | 602 | 602 | 0 | 0 |
| test1 | 1930 | E2 | 65.11 | 602 | 364 | 187 | 154 | 2747 |
| test2 | 1176 | E0 | 55.89 | 257 | 257 | 257 | 0 | 0 |
| test2 | 1176 | E1 | 62.49 | 314 | 314 | 314 | 0 | 0 |
| test2 | 1176 | E2 | 61.45 | 314 | 156 | 101 | 49 | 1283 |

## Interpretation

- E1 的低置信度/NMS 调参增加了两段视频的 Track interruptions，不提升为默认检测参数。
- E2 没有减少检测器本身的 interruption，但逻辑实体层显著降低了实体碎片化：test1 从 469 降至 187（约 60.1%），test2 从 257 降至 101（约 60.7%）。Raw fragmentation 分别下降约 22.4% 和 39.3%。
- E2 在这两段 RTX 5060 Ti 运行中没有出现 FPS 下降：test1 为 65.11 FPS，test2 为 61.45 FPS；该比较同时改变了检测阈值，不能作为纯粹的预测开销隔离实验。
- 预测框只用于短时显示连续性，不进入 ByteTrack、不更新球队原型，也不作为越位/犯规等高风险判定依据。
- 本基准没有运行球队分类模型和人工标签准确率评测，因此不能由这些数字推出球队标签准确率；球队分队逻辑仍保持赛前监督标定与 UNKNOWN 拒绝路径。
