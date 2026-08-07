# Tracking stability benchmark

日期：2026-08-06

设备：本地 CPU（型号未记录）

视频窗口：`test1.mp4` 和 `test2.mp4` 各前 300 帧，`imgsz=640`。

说明：没有人工逐帧 ID ground truth，因此本报告使用生产链路直接记录的生命周期指标，不声称 IDF1/HOTA。

## Profile

| Profile | 检测 conf | NMS IoU | lost buffer | 短时预测 | 实体重连 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 current | 0.25 | 0.70 | 45 | 0 帧 | 关闭 |
| E1 tuned | 0.20 | 0.80 | 60 | 0 帧 | 关闭 |
| E2 prediction + entity | 0.20 | 0.80 | 60 | 6 帧 | 12 帧窗口 |

## Results

| 视频 | Profile | FPS | Track interruptions | Raw fragmentations | Entity fragmentations | ID switches | Predicted frames |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test1 | E0 | 7.03 | 38 | 38 | 38 | 0 | 0 |
| test1 | E1 | 7.21 | 61 | 61 | 61 | 0 | 0 |
| test1 | E2 | 7.24 | 61 | 29 | 18 | 7 | 261 |
| test2 | E0 | 7.31 | 31 | 31 | 31 | 0 | 0 |
| test2 | E1 | 6.99 | 53 | 53 | 53 | 0 | 0 |
| test2 | E2 | 6.96 | 53 | 27 | 22 | 1 | 226 |

## Interpretation

- E1 的低置信度/NMS 调参在这两个窗口中增加了 raw fragmentation，因此不提升为默认检测参数。
- E2 没有减少检测器本身的 interruption，但通过 6 帧显示预测和逻辑实体重连降低了实体层 fragmentation；test2 从 31 降到 22，test1 从 38 降到 18。
- E2 的预测框只用于短时显示连续性，不进入 ByteTrack、不更新球队原型，也不作为越位/犯规等高风险判定依据。
- 这组 CPU 窗口用于回归和 profile 选择，不等价于 RTX 5060 Ti 性能结果。CUDA 基准需要远程主机恢复 NVIDIA 设备、Torch 环境和视频文件后重新运行。
