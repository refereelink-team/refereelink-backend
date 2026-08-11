# 球场感知部署与长时间验收

本页定义 Constrained Pitch Tracker 的阶段 G 交付边界。模型导出、性能验证和准确率验证是三件不同的事，报告中不得互相替代。

## 1. 能力与依赖

PyTorch/CUDA 是当前可运行基线。ONNX 导出依赖独立的可选组：

```bash
uv sync --extra export
```

TensorRT 不作为 Python 项目的隐式依赖。目标机器必须安装与 CUDA/驱动兼容的 NVIDIA TensorRT SDK，并能在 `PATH` 中找到 `trtexec`。缺少这些能力时，导出命令会明确失败并打印能力报告，不会生成伪 `.engine` 文件。

## 2. 导出训练权重

仅接受训练脚本生成、包含验证信息的 V1 checkpoint：

```bash
uv run python tools/export_pitch_perception.py \
  --checkpoint runs/pitch/c1-best.pt \
  --output runs/pitch/c1.onnx \
  --format onnx \
  --report runs/pitch/c1-onnx-report.json
```

RTX 目标机上的 TensorRT FP16：

```bash
uv run python tools/export_pitch_perception.py \
  --checkpoint runs/pitch/c1-best.pt \
  --output runs/pitch/c1-fp16.engine \
  --format tensorrt \
  --device cuda \
  --fp16 \
  --report runs/pitch/c1-tensorrt-report.json
```

导出脚本会执行 ONNX checker；TensorRT 构建交由官方 `trtexec` 完成。生产验收仍需用同一测试集比较 PyTorch、ONNX 与 TensorRT 的输出误差。

在目标机上先执行确定性输入的数值一致性和运行时检查：

```bash
uv run python tools/validate_pitch_perception_deployment.py \
  --checkpoint runs/pitch/c1-best.pt \
  --model runs/pitch/c1.onnx \
  --device cuda:0 \
  --provider CUDAExecutionProvider \
  --precision fp16 \
  --report runs/pitch/c1-onnx-cuda-report.json
```

工具会拒绝静默回退到 CPU，并分别记录同步后的 PyTorch forward 延迟和包含传输/输出物化的 ONNX Runtime `session.run` 延迟。该命令仍固定声明 `deployment_accuracy_valid=false`；数值接近只说明导出未明显改变网络输出，不能替代带真值测试集的准确率评估。

## 3. 长时间 CUDA 联调

以下命令循环读取本地视频，默认连续运行 30 分钟：

```bash
uv run python tools/benchmark_pitch_registration_long_run.py \
  --source test2.mp4 \
  --device cuda \
  --duration-minutes 30 \
  --output docs/pitch-registration-long-run.json
```

开发阶段可先限制帧数做 smoke test：

```bash
uv run python tools/benchmark_pitch_registration_long_run.py \
  --source test2.mp4 \
  --device cuda \
  --duration-minutes 30 \
  --max-frames 300 \
  --output /tmp/pitch-registration-smoke.json
```

报告包含：

- 端到端 FPS 与帧延迟 median/P95/max；
- 相机状态分布、语义模型调用次数和复用率；
- 接触点坐标可用率与 `sigma_m` 分布；
- RSS 首尾差和每分钟线性斜率；
- CUDA 峰值显存。

该报告没有真值，因此固定写入 `accuracy_valid=false`。它只能证明性能和稳定性；JaC、线误差、网格米制误差与球员坐标误差必须由独立人工真值报告给出。

默认先预热 30 帧，预热不进入延迟与显存统计。少于 5 分钟的 smoke run 不报告 RSS 线性斜率，避免把模型和分配器冷启动误判为持续内存泄漏。

延迟和坐标 σ 分位数使用固定容量、固定随机种子的 reservoir sample，最大值单独精确记录。这样长跑工具自身不会因无限追加观测而制造线性 RSS 增长；JSON 同时输出总观测数和实际样本数。

未传入 `--camera-rig-profile-path` 时，V2 只能依赖通用单应性重定位，报告会明确写入 `physical_pan_constraint_active=false`。这种运行可用于兼容性和吞吐测试，不能作为固定云台生产几何验收；后者必须加载由真实多 pan 锚帧生成并通过门禁的 rig profile。

## 4. 验收门槛

在选定生产模型前，至少满足：

- 同一真实测试集上，部署后输出与 PyTorch 基线的数值差异在预设容差内；
- RTX 5060 Ti 上完整管线 FPS 相比当前生产基线下降不超过 10%；
- 连续 30–60 分钟无崩溃、无持续显存增长；
- `PREDICTED/LOST` 相机状态不会被越位、犯规定位或距离统计消费；
- 所有失败报告保留视频时间段、相机状态和误差指标，便于复现。
