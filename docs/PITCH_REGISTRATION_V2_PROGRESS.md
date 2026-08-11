# Constrained Pitch Tracker 开发进展

更新日期：2026-08-11
开发分支：`codex/dev-pitch-projection`

## 已完成

### 核心几何与状态链路

- 公制 105×68 m 球场模型、统一 32 点词表和语义线/圆弧几何；
- pinhole/fisheye 标定文件、运行时去畸变和分辨率缩放；
- `CameraRigProfile`、固定机位水平 pan 投影模型和多锚帧安装标定；
- USAC MAGSAC 初始化、退化检查、点线 pan-only 鲁棒优化；
- 动态区域排除的稀疏 LK、前后向检查、pan EKF 和自适应语义刷新；
- `INITIALIZING/RELOCALIZED/CORRECTED/TRACKED/PREDICTED/LOST` 状态；
- 接触点 head/ankle/bbox 安全回退接口和投影协方差传播；
- 以配置开关接入 `VisionCore`，默认兼容旧 32 点模型。

### 真值与评估工具

- 从视频按时间范围均匀抽取、按序列指定 split 的标注包生成器；
- 无构建依赖的浏览器标注页：球场锚点、球员接触点、翻帧、草稿和 JSON 导出；
- 标注格式、文件、标签、点数、覆盖度、退化和人工拟合残差验证；
- 支持稀疏对应点、稠密真值矩阵、接触点和延迟的评估器；
- 设计稿和 1536×1024 真实 `test1.mp4` 浏览器实现截图。

### 镜头与固定云台标定

- `camera.npz` V2 保存训练 RMS、留出 median/P95 和有效视图数，同时兼容 V1；
- pinhole/fisheye 按棋盘格视图留出验证，不再只比较训练 RMS；
- 少于 6 个有效视图时禁止 `auto` 猜测模型；
- 多 pan 人工锚帧统一在去畸变坐标系中生成 `CameraRigProfile`；
- 相机中心离散和锚帧重投影误差不通过门禁时不保存运行 profile。

### 球场感知候选 C1

- 固化 20 类语义线/圆弧、33 个 landmark 和 offset 的统一输出契约；
- 实现 MobileNetV3-Large 共享 backbone 双 head、数据集、损失、训练脚本和 PyTorch 推理适配；
- 亚像素 heatmap 解码已消费 offset head；
- RTX 5060 Ti 512×288 FP16：mean 4.29 ms，P95 6.82 ms，峰值 48.45 MB；
- 上述为随机权重结构延迟，准确率仍待真实训练/测试，未选定生产模型。

### 地面接触点与安全投影接口

- `VisionCore` 在 V2 路径中使用统一接触点选择器，当前按 bbox bottom-center 回退；
- 画面边缘截断、小框和异常框不会被强制投影；
- 公制坐标、图像误差传播后的 `sigma_m`、接触点来源和相机状态随球员输出；
- 旧 `field_x/field_y` 厘米接口继续保留，新接口显式输出 `field_x_m/field_y_m`；
- `PREDICTED/LOST` 或投影不确定性超限时 `field_coordinate_usable=false`；
- 控球距离和越位候选只消费通过门禁的球员坐标。

### 部署与长时间验收工具

- 增加训练 checkpoint 的 ONNX checker 导出和 `trtexec` TensorRT 构建入口；
- 增加 PyTorch/ONNX Runtime 数值一致性与延迟检查，拒绝静默 CPU fallback；
- 缺少 ONNX/TensorRT 时输出明确能力错误，不生成替代模型；
- 增加循环视频的 30–60 分钟 CUDA 稳定性工具，记录延迟、RSS 斜率、峰值显存、相机状态和安全坐标率；
- 增加可执行的 soak 报告门禁：时长、RSS 斜率、峰值显存、FPS 降幅、P95 延迟和物理 pan profile 均可独立约束；
- 性能报告固定声明 `accuracy_valid=false`，不以无真值运行替代准确率评估。

## 验证结果

### 自动化测试

- CPT 首次实现提交前：本地 `163 passed`，远端 CUDA 环境 `163 passed`；
- 标注与评估增量：`24 passed`；
- 当前完整回归：本地与远端 `14186de` 隔离快照均为 `201 passed`、11 warnings；
- 本次修改文件 Ruff 和 Web TypeScript/Vite build 通过；
- 浏览器实测：真实 `test1.mp4` 8 帧包，4 锚点、1 接触点、草稿恢复、翻帧和导出反馈通过；
- 响应式：1536×1024、900 px 和 430 px 宽度验证，无横向溢出。
- 远端已生成按序列隔离的 `test1-calibration` 与 `test2-test` 空真值包，各 30 个可解码帧并通过结构校验；人工锚点和接触点仍待填写；
- 修复 MP4 容器高报尾部帧数、尾部随机 seek 失败及多个请求回退到同一帧的问题，清单始终记录真实解码帧号。

### RTX 5060 Ti 兼容链路

| 视频 | 帧数 | 端到端 FPS | 球员检测 | 球场模型/次 | 模型调用 | 复用率 | 单应性可用率 | GPU allocated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test1 | 300 | 36.28 | 20.39 ms | 14.57 ms | 75 | 75.0% | 100% | 194.1 MB |
| test2 | 300 | 41.24 | 17.87 ms | 14.33 ms | 64 | 78.7% | 100% | 194.1 MB |

同机旧版 `test1` 为 36.88 FPS；兼容接入后约下降 1.6%，低于 10% 性能预算。

上述结果只能证明当前兼容路径的速度和矩阵可用性，不能证明投影准确。真实 P95 线误差、网格米制误差和球员接触点误差必须等待独立真值标注后填写。

### C1 ONNX CUDA 部署链路

- FP16 ONNX checker 与 ONNX Runtime 1.28 CUDA provider 通过；
- 三个输出头相对 PyTorch 的最大绝对误差不超过 `1.221e-4`；
- 512×288、batch=1：PyTorch FP16 P95 `5.99 ms`，ONNX Runtime `session.run` P95 `4.89 ms`；
- checkpoint 为明确标记的随机权重，仅验证导出/运行时，不构成准确率结果；
- 目标机缺少 `trtexec`，TensorRT engine 尚未验证。

### 30 分钟 CUDA soak

- `test2.mp4` 循环 86 次，共 101,212 帧，56.23 FPS；
- 帧延迟 median `13.04 ms`、P95 `30.16 ms`，峰值显存 `222.41 MB`；
- RSS 30 分钟增长 `38.66 MB`，线性斜率 `0.90 MB/min`；
- 无 rig 稳定性门禁通过；物理 pan 生产门禁未执行且不应视为通过；
- 安全坐标可用率 `27.66%`，反映无 rig 时 PREDICTED/LOST 坐标被正确拒绝；
- 发现并修复 Ultralytics `half` 参数逐帧输出弃用日志，20 帧远端 CUDA 复测不再重复打印。

## 尚未完成

1. `test1/test2/真实云台` 的人工真值和至少 200 个接触点尚未完成；
2. pinhole/fisheye 代码已支持留出验证，但还缺同一真实相机数据和边缘直线复核；
3. 尚无真实固定机位的多 pan 锚帧，因此物理 pan-only 链路目前只有合成几何测试；
4. MobileNetV3-LR-ASPP、PIDNet-S、SegFormer-B0 尚未在同数据上训练比较；
5. 接触点神经 head 尚未训练，当前已接入 bbox 安全回退、米制不确定性和下游门禁；
6. 随机权重 C1 的 ONNX CUDA 链路和无 rig 30 分钟稳定性报告已完成；训练后权重的准确率、TensorRT engine 与真实 rig 长跑仍未完成。

## 下一里程碑

阶段 A 的代码工具已完成；数据完成条件仍需人工标注。开发侧下一步是：

人工采集、标注、复核和交付流程见 `docs/PITCH_REGISTRATION_MANUAL_DATA_GUIDE.md`。

1. 完成镜头留出集模型选择和多 pan 锚帧 profile 工具；
2. 固化球场感知训练数据契约、双 head 基线和模型竞赛脚本；
3. 在人工真值到位后冻结 P0 准确率基线，再决定 C1/C2/C3 的生产模型；
4. 标注接触点训练/测试集，训练单接触点 head 并与 bbox 回退做误差和速度尖峰对照。
