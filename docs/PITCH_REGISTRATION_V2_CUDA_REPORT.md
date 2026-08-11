# Constrained Pitch Tracker CUDA 验证报告

更新日期：2026-08-11  
分支：`codex/dev-pitch-projection`  
验证提交：`0a97168`  
设备：NVIDIA GeForce RTX 5060 Ti 16 GB

## 1. 结论边界

当前结果证明：

- V2 接口、旧 32 点适配器、人员检测、ByteTrack 和安全坐标门禁可以在 CUDA 环境连续联调；
- C1 MobileNetV3 双 head 可以导出 FP16 ONNX，并由 ONNX Runtime CUDA provider 执行；
- 导出模型三个输出头与 PyTorch FP16 输出满足当前数值容差；
- 自动化回归在本地和远端隔离快照均通过。

当前结果**不能证明**：

- 随机权重 C1 具有球场感知准确率；
- 未加载真实 `CameraRigProfile` 的通用单应性路径满足固定云台生产几何；
- TensorRT engine 已构建或验证；
- JaC、线重投影误差、球员米制误差达到目标。

上述准确率结论必须等待阶段 A 的独立人工真值，以及阶段 B 的真实镜头和多 pan 安装标定数据。

## 2. 自动化回归

| 环境 | 结果 |
|---|---:|
| macOS 本地工作区 | 201 passed，11 warnings |
| RTX 5060 Ti 远端 `14186de` 隔离快照 | 201 passed，11 warnings |
| 本次修改文件 Ruff | passed |
| Web TypeScript/Vite build | passed |

现有 warning 包含 Supervision `KeyPoints.confidence` 弃用提示和 ByteTrack 弃用提示，均未导致本轮失败；后续依赖升级前应迁移到新接口。

## 3. C1 FP16 ONNX 部署检查

输入为 `1×3×288×512`，checkpoint 明确标记：

```text
accuracy_valid = false
purpose = random_export_smoke_only
```

### 3.1 能力

| 项目 | 结果 |
|---|---|
| PyTorch | 2.13.0+cu130 |
| ONNX Runtime | 1.28.0 |
| ONNX checker | passed |
| CUDAExecutionProvider | active |
| TensorRT `trtexec` | unavailable |

ONNX Runtime 包枚举到 `TensorrtExecutionProvider` 不等于 TensorRT SDK 可用于构建生产 engine。当前缺少 `trtexec` 和 TensorRT Python SDK，因此 TensorRT 验收仍未完成。

### 3.2 数值一致性

| 输出 | Shape | 最大绝对误差 | 平均绝对误差 | atol=0.02, rtol=0.02 |
|---|---|---:|---:|---|
| semantic logits | 1×21×288×512 | 0.0001221 | 0.00000810 | passed |
| landmark heatmaps | 1×33×72×128 | 0.0000610 | 0.00000048 | passed |
| landmark offsets | 1×66×72×128 | 0.0000610 | 0.00000042 | passed |

### 3.3 延迟

30 次预热、150 次测量：

| Runtime | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| PyTorch FP16 forward + CUDA sync | 3.54 ms | 2.96 ms | 5.99 ms | 6.72 ms |
| ONNX Runtime CUDA `session.run` | 4.19 ms | 4.13 ms | 4.89 ms | 6.08 ms |

两行不是完全相同的计时边界：PyTorch 输入已在 GPU，ONNX Runtime `session.run` 包含 host/device 传输和 CPU 输出物化。该结果用于确认部署预算和异常回退，不用于宣称某个 runtime 绝对更快。

## 4. V2 通用几何 smoke

以下运行使用旧 32 点模型适配器，没有真实固定云台 rig profile。它只能检查集成速度和安全门禁。

| 视频 | 测量帧 | FPS | P95 帧延迟 | 坐标可用率 | σ median | σ P95 |
|---|---:|---:|---:|---:|---:|---:|
| test1 | 300 | 63.50 | 27.16 ms | 27.59% | 0.531 m | 0.866 m |
| test2 | 300 | 38.04 | 29.03 ms | 32.80% | 0.541 m | 0.759 m |

坐标可用率低的主要原因是：无 rig profile 时不能执行受物理约束的逐帧 pan 光流更新；语义关键帧之间的状态为 `PREDICTED`，安全门禁按设计拒绝将这些帧用于越位、距离和控球事件。不能通过放宽门禁把该问题伪装成高可用率。

## 5. 长时间运行

`test2.mp4` 循环 30 分钟，预热 30 帧不进入统计：

| 指标 | 结果 |
|---|---:|
| 测量时长 | 1800.014 s |
| 总帧数 / 视频循环 | 101,212 / 86 |
| 端到端 FPS | 56.23 |
| 帧延迟 median / P95 / max | 13.04 / 30.16 / 158.58 ms |
| CUDA peak allocated | 222.41 MB |
| RSS first / last / growth | 2167.71 / 2206.38 / 38.66 MB |
| RSS 线性斜率 | 0.90 MB/min |
| 单应性可用率 | 96.94% |
| 语义推理次数 / 复用率 | 40,188 / 60.29% |
| 安全坐标可用率 | 27.66% |
| 坐标 σ median / P95 / max | 0.549 / 1.076 / 13.726 m |

相机状态分布：

| 状态 | 帧数 | 占比 |
|---|---:|---:|
| RELOCALIZED | 86 | 0.08% |
| CORRECTED | 33,566 | 33.16% |
| PREDICTED | 64,464 | 63.69% |
| LOST | 3,096 | 3.06% |

以 test2 兼容基线 `41.24 FPS`、最大 FPS 降幅 10%、P95 35 ms、RSS 5 MB/min 和峰值显存 500 MB 运行自动门禁，结果为 `valid=true`、无 issue。该通过项只代表**无 rig 的性能/稳定性 smoke**；没有使用 `--require-physical-pan`，因此不构成固定云台生产验收。

长跑使用固定容量 reservoir 保存延迟和坐标 σ 分布，避免基准工具自身因无限追加样本造成线性内存增长。

本次长跑还发现 Ultralytics 8.4.95 会对旧 `half` 参数逐帧打印弃用警告，使 stdout 增长到约 13 MB。代码已改为当前 `quantize=16` 参数，并保留旧 fake/接口 fallback；RTX 5060 Ti 上重新运行 20 帧后，每帧警告消失，只剩一次 ByteTrack 未来弃用提示。

## 6. 当前决策

1. 保留 V2 安全门禁，不把无观测的 `PREDICTED` 坐标提升为可信坐标；
2. C1 已满足结构延迟预算，但没有真实训练结果，不能选为生产模型；
3. 没有阶段 A 真值前不启动 C1/C2/C3 大规模训练，避免把镜头、几何和感知误差混在一起；
4. 真实固定云台必须完成镜头模型选择和多 pan rig profile 后，再验收 `TRACKED` 覆盖率；
5. TensorRT 在目标机安装官方 SDK/`trtexec` 后再执行独立 engine 验收。
