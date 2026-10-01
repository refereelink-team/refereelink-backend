# Foul Candidate Detection Trade-offs

## 实现方式

`app/foul_detection/detector.py` 通过可插拔的 `FoulPredictor` 协议调用推理。
默认实现 `MViTFoulPredictor` 使用 SoccerNet VARS MViT V2 Small 权重
(`assets/weights/14_model.pth.tar`)。

## 替换 offside 适配层的原因

原实现依赖外部 `offside.foul_model`（来自 `fouls_far` 包），该适配层从未
完成、也从未随仓库发布。本 PR 移除该依赖，改为自包含的 predictor 实现。

## 防噪机制

1. **置信度阈值**（默认 0.5）：丢弃弱预测。
2. **冷却帧数**（默认 25 帧）：避免相邻帧重复报警。

单元测试覆盖：
- `test_drops_noise_on_ten_second_clip`
- `test_avoids_zero_detections_on_foul_clip`
- `test_single_foul_clip_yields_exactly_one_candidate`
- `test_weak_inference_output_is_filtered`

## 真实视频验证（已执行）

环境：NVIDIA GeForce RTX 5060 Ti，CUDA 可用。
输入：4.8 秒犯规视频（145 帧 @ 30fps）。
参数：`cooldown_frames=200`。
结果：
- 输出候选：1 个
- 动作分类：Elbowing
- 严重度：Offence + Yellow Card
- 置信度：0.615
- 推理次数：16

## 已知局限性

- MViT V2 Small 训练时使用 4 个机位；本模块将单机位输入复制到 4 个注意力槽
  作为临时方案，准确率可能低于多机位场景。
- 单次推理约 360ms，高于实时要求。后续可替换为更轻量的单机位模型。
- 本 PR 只验证了单个犯规片段的输出数量，未做完整数据集准确率评估。
- 犯规位置投影（radar 图上的标记）暂未实现，`foul_location` 恒为 `None`。

## 审查修复与验收边界

- `No Offence` 不再输出候选，也不会消耗报警冷却时间；注入 predictor 和事件适配层同样忽略明确的无犯规结果。
- 默认权重路径统一为 `assets/weights/14_model.pth.tar`，与多视角模块及 `tools/setup_mvfoul.sh` 一致。
- `FOUL_DETECTION`、`RADAR` 和旧雷达桌面模式在视频上保留两秒候选提示；Web 提示保留五秒，其他事件不会取消隐藏计时器。
- 新增无犯规 logits、冷却、默认权重路径以及独立视频提示的回归测试。这些测试使用模拟模型，不作为真实视频准确率证据。
- 上述单片段 GPU 结果是原实现的记录，修复后仍需用实际权重重新测量正常片段、代表性犯规片段、误报、漏报及端到端延迟。
- 轻量模型对比和降级尚未完成；约 360ms 的同步推理仍可能阻塞主流水线，不能据此宣布 Issue #69 的实时性验收完成。
