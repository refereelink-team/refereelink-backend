# 广播视频与低成本云台球场投影实现报告

更新日期：2026-08-13

分支：`codex/dev-pitch-projection`
目标设备：NVIDIA GeForce RTX 5060 Ti

## 1. 当前结论

工程链路已经完成，但生产模型尚未通过准确率验收。

已完成：

```text
公开教师资产审计与缓存协议
→ SoccerNet 26 类读取和 20 类部署映射
→ MobileNetV3 双 head 监督/蒸馏/域适配训练
→ checkpoint/ONNX/CUDA 推理
→ 广播 Shot、Pan/Tilt/Zoom EKF、无 Rig 光流
→ 点线联合优化与三档安全状态
→ 离线双向跟踪和 RTS 平滑
→ 准确率、性能、消融自动门禁
```

未完成的不是接口，而是真实数据工作：SoccerNet 尚未下载训练，且
`test1/test2` 尚无独立的 40–60 帧投影真值。因此当前不能选择生产学生
权重，也不能宣称米制准确率达标。

## 2. 模式和运行边界

| 模式 | 相机状态 | 用途 |
|---|---|---|
| `LEGACY` | 旧单应性复用 | 兼容旧入口 |
| `BROADCAST` | Shot 内 pan/tilt/roll/focal + 光流/EKF | 广播视频和可变焦云台 |
| `RIG_PAN` | 固定相机中心、内参和 tilt，只估 pan | 已完成安装标定的低成本左右云台 |

`SAFE` 才能进入高风险事件逻辑；`PREVIEW` 只允许视频调试和展示；
`UNAVAILABLE` 不输出投影。硬切会清空 Shot 状态，禁止跨镜头复用。

学生 checkpoint 现在可从 CLI/API 注入：

```bash
uv run python -m app.server.main \
  --video_source test2.mp4 \
  --device cuda \
  --field_registration_mode broadcast \
  --pitch_perception_checkpoint_path runs/pitch/mobilenet-v3-dual-head.pt
```

不传 checkpoint 时继续使用旧 `football-pitch-detection.pt` 适配器。
V2 默认球场与 SoccerNet/教师统一为 105×68 m；兼容的厘米坐标、边界过滤
和 Radar 绘制都从同一个 `PitchDimensions` 构造，避免旧 120×70 画布混入。

## 3. 轻量模型

部署候选为 3,253,384 参数的 MobileNetV3-Large 双 head：

```text
512×288 RGB
  ├── 21 类语义 logits（20 个场地元素 + background）
  ├── 33 个 landmark heatmap
  └── 33×2 亚像素 offset
```

RTX 5060 Ti 合成训练 smoke 后的真实 checkpoint 推理结果：

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| FP16 P95 | 6.36 ms | ≤ 8 ms |
| 峰值 allocated | 48.45 MB | ≤ 300 MB |
| 参数量 | 3.25 M | 信息项 |

这些数字只证明结构轻量。合成 fixture 的 IoU/PCK 不是准确率证据。

## 4. 教师与数据决策

### PnLCalib

- 固定提交：`8c87391d6f4ea40c5e4d65e61529916c7a49ce62`；
- GPL-2.0，只在独立研究环境执行；
- SV 权重总计约 506 MiB，哈希已写入资产登记表；
- 12 个本地代表帧中，默认阈值返回 7 个相机解；抽查存在明显错误投影；
- 双 HRNet 前向 P95 113.89 ms，完整 PnL P95 134.49 ms。

### TVCalib

- 固定提交：`1222c5230af2742395d74918ed6f34eb2b9bf7f9`；
- MIT；checkpoint 哈希已登记；
- `test2` 第 480 帧语义线可视结果合理；
- 256×454 输入 P95 16.73 ms。

因此 PnLCalib 是相机/点候选教师，TVCalib 是语义线交叉教师。只有二者
一致且几何门禁通过的帧才允许进入蒸馏缓存。教师都不进入实时运行时。

### SoccerNet

转换器保留官方 26 类原始词表，并将 20 类地面线/圆弧映射到部署头；
不可见 landmark 不参与 loss；split 检查按比赛/序列防止泄漏。数据和派生
权重受研究用途约束，不提交 Git。

## 5. 当前真实视频结果

旧 32 点模型在 `test1/test2` 两遍离线处理后的最终安全结果：

| 视频 | 实际解码帧 | 两遍吞吐 | SAFE | PREVIEW |
|---|---:|---:|---:|---:|
| test1 | 1,930 | 13.90 FPS | 0 | 1,930 |
| test2 | 1,176 | 14.44 FPS | 0 | 1,176 |

早期点拟合曾把错误矩阵标成 SAFE。视觉检查后，代码增加了物理相机与
独立点线证据双门禁，所有旧模型结果因此降为 PREVIEW。这说明安全逻辑
生效，也说明现有感知模型不适合作为最终投影模型。

结果视频：

- `/tmp/test1-full-pitch-105x68.mp4`
- `/tmp/test2-full-pitch-105x68.mp4`

视频中的橙色表示 PREVIEW，不应理解为准确坐标。

## 6. 评测工具

生成离线结果：

```bash
uv run python tools/run_offline_pitch_registration.py \
  --source test2.mp4 \
  --pitch-perception-checkpoint runs/pitch/mobilenet-v3-dual-head.pt \
  --device cuda \
  --output-video /tmp/test2-field.mp4 \
  --output-registration /tmp/test2-field \
  --report /tmp/test2-field-performance.json
```

对 held-out 标注计算误差并执行硬门禁：

```bash
uv run python tools/evaluate_offline_pitch_registration.py \
  --annotations annotations/test2/manifest.json \
  --registration /tmp/test2-field \
  --output reports/E6-accuracy.json

uv run python tools/check_pitch_registration_accuracy.py \
  reports/E6-accuracy.json
```

性能和模型选择：

```bash
uv run python tools/check_pitch_registration_performance.py \
  reports/E6-performance.json

uv run python tools/summarize_pitch_registration_ablation.py \
  --report-root reports \
  --output reports/summary.json \
  --teacher-jac-at-5 0.0 \
  --student-jac-at-5 0.0
```

门禁工具会因缺少报告、`accuracy_valid=false`、SAFE 覆盖不足、错误 SAFE、
30 分钟稳定性不足或性能超预算而返回非零退出码。

完整性能报告由长跑工具通过 `--model-benchmark-report` 合入模型 P95/增量
显存，并通过 `--baseline-fps` 写入同设备基线；因此模型单独推理达标不会
掩盖完整管线 FPS 或长时间内存稳定性失败。

## 7. 下一次可执行工作

1. 获取 SoccerNet-Calibration，分别建立官方 train/validation/test index；
2. 在 CUDA 设备执行 30 epoch 监督训练、10 epoch 双教师蒸馏、5 epoch
   `test1` 共识伪标签域适配；
3. `test2` 完全不参与训练或阈值选择；
4. 对两个本地视频各标 40–60 个 held-out 帧，运行 E0–E6；
5. MobileNet 达标即冻结；只有 JaC@5 与教师差距超过 7 个百分点且完整
   门禁失败，才解锁 PIDNet-S。

在上述数据到位之前，不应通过调低阈值把 PREVIEW 升级为 SAFE，也不应
训练或发布 PIDNet/SegFormer 来掩盖缺少真值的问题。

## 8. 工程验证状态

- 本地与远端 Python 回归测试全部通过；
- Web TypeScript/Vite production build 通过；
- 本阶段 field 相关脚本可从任意 checkout 直接运行，不会误导入另一份
  editable 安装；
- RTX 5060 Ti 上完成真实 checkpoint 加载与 FP16 推理 smoke；
- 当前准确率门禁仍失败，原因是没有真实学生权重和 held-out 投影真值，
  不是通过降级阈值规避。
