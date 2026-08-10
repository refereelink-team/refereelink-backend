# 球场关键点识别与 2D 投影 V2 设计

> 调研日期：2026-08-10
>
> 开发分支：`codex/dev-pitch-projection`
>
> 范围：单个固定安装、固定焦距、只允许水平旋转的广角云台相机；同时兼容测试用的主转播机位片段。

## 1. 结论

推荐重构为一条 **受约束的点线融合时序标定链路**：

```text
离线相机内参/畸变标定
  ↓
自适应关键帧：轻量语义线段 + 稀疏关键点检测
  ↓
MAGSAC++ 初始解 + 点线联合非线性优化
  ↓
固定机位/固定焦距/水平云台物理约束
  ↓
逐帧光流传播 + 1D pan EKF
  ↓
带置信度的 CameraState 与 image↔pitch 变换
  ↓
球员地面接触点检测/回退
  ↓
球场坐标、误差估计和安全门禁
```

该方案在文档中简称 **Constrained Pitch Tracker（CPT）**。它不是把现有 32 点检测器简单升级，而是把系统主状态从“每隔几帧重新猜一个 8 自由度单应性”改为“先建立物理相机，再持续跟踪水平转角”。

选择它的核心原因：

1. **任务匹配性最高**：相机位置、焦距、俯仰和横滚在比赛中固定，动态量主要是水平转角；运行时没有必要反复估计任意 8-DoF 单应性。
2. **线段比交点更常见**：画面可能看不到 4 个可靠交点，但通常仍能看到边线、中线、禁区线或圆弧的一部分。
3. **时序信息没有被浪费**：相邻视频帧高度相关，光流和状态滤波比“复用旧矩阵”更能跟随真实云台运动。
4. **轻量化来自降维**：高质量检测只在关键帧执行，普通帧只做稀疏光流和 1D 状态更新。
5. **可审计**：每一帧都能输出点误差、线误差、观测覆盖度、状态协方差和重定位原因。

## 2. 当前实现审计

当前链路位于：

- `app/vision/core.py`
- `app/geometry/pitch_projection.py`
- `app/geometry/camera.py`
- `app/config/pitch.py`

当前行为：

- 球场模型输出 32 个预定义关键点；
- 置信度低于 `0.35`、靠近图像边缘的点被过滤；
- 至少 4 点后依次尝试 `RANSAC → RHO → LMEDS`；
- 使用固定 5 px RANSAC 阈值、12 px 平均重投影误差门限；
- 默认每 5 帧运行一次球场模型；
- 其余帧直接复用上一矩阵，累计最多 0.5 秒；
- 相机移动使用低分辨率 phase correlation 检测全局平移；
- 球员坐标使用检测框 `BOTTOM_CENTER` 经过单应性变换。

现有 CUDA 基准显示球场模型在 640 输入时单次推理约 14 ms，每 5 帧运行时平均摊销约 2.8 ms。这个速度已经不是主要矛盾；更大的问题是现有指标只记录 `homography_available_ratio`，不能说明矩阵是否准确、是否抖动或是否把球员投影到了正确位置。

### 2.1 当前方案的结构性问题

1. **4 个点不等于几何充分**：4 点可能接近共线、集中在画面一角，仍会产生病态且外推误差很大的单应性。
2. **关键点稀疏**：球员遮挡、白线磨损、画面裁切或只看到一条边线时容易失效。
3. **没有使用线与圆弧的全部像素证据**：大量可用于约束相机的球场标线被压缩成少数交点。
4. **固定像素阈值不随分辨率和不确定度变化**。
5. **phase correlation 只近似平移**：云台水平旋转形成的图像变化并非严格二维平移，前景球员和观众也会污染估计。
6. **旧矩阵复用不是运动估计**：相机已经转动时，旧矩阵即使状态为 `REUSED` 也可能产生系统性坐标偏差。
7. **单应性是无约束状态**：固定机位只需少量物理参数，8-DoF 容易出现非物理跳变。
8. **畸变模型未做选择验证**：当前是普通针孔畸变；广角相机可能更适合 fisheye 模型。畸变残差会在画面边缘放大到场地坐标。
9. **球场尺寸默认为 120×70 m**：若真实场地为其他尺寸，会直接造成速度、距离和位置尺度误差。必须使用场馆实际尺寸，或明确输出为规范化坐标。
10. **检测框底部中心不是稳定的地面接触点**：遮挡、截断、跑步姿态和不完整框都会产生偏差。

## 3. 技术路线调查与取舍

| 路线 | 优点 | 主要问题 | 轻量化 | 任务匹配 | 结论 |
|---|---|---|---:|---:|---|
| 32 点 + RANSAC | 简单、已有实时实现 | 点稀疏、抖动、退化不可控 | 高 | 中 | 仅作为 E0 基线 |
| KpSFR / 均匀网格关键点 | 增加可见点概率，可直接 DLT | 仍依赖点；不充分利用长直线/圆弧 | 中 | 中 | 可作初始化对照 |
| TVCalib | 直接利用语义线段和物理相机参数 | 多初值与迭代优化较重 | 低至中 | 高 | 借鉴损失和验证，不整套照搬 |
| PnLCalib | 点线互补，单帧精度强 | 官方 HRNet-W48 双网络较重，论文配置约 164 ms 起 | 低 | 高 | 借鉴初始化与点线 refinement |
| TacticalCalib | 直接回归 6-DoF，对遮挡有鲁棒性 | 依赖训练分布；仍有不完整输出；缺少强时序约束 | 中 | 中 | 只做初始化候选，不作主线 |
| BHITK | 显式建模关键点和单应性的时序不确定度 | 状态较大，仍建立在关键点观测上 | 高 | 高 | 借鉴协方差、滤波和门控 |
| AuxFlow | 锚帧 + 光流辅助点，直接面向视频稳定性 | 2026 方法，工程实现和权重成熟度需验证 | 高 | 很高 | 借鉴锚帧传播机制 |
| BroadTrack | 语义标线、光流、镜头畸变、云台物理约束、重定位完整 | 原版在双 RTX 4090 的 HD 输入约 16 FPS | 低 | **最高** | 作为架构母版，重新做轻量版本 |

研究证据：

- [KpSFR](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.html) 用均匀分布关键点增加不同视角下的可见对应点。
- [TVCalib](https://arxiv.org/abs/2207.11709) 证明可以直接利用带语义的线/圆弧像素，通过段重投影损失优化物理相机参数。
- [PnLCalib](https://github.com/mguti97/PnLCalib) 使用关键点获得初值，再用球场线进行点线联合 refinement；官方实现使用两套 HRNetV2-W48 网络，因此不适合作为本项目逐帧运行的原样方案。
- [BHITK](https://arxiv.org/abs/2311.10361) 显式建模相邻帧变换和关键点不确定度，说明时序滤波能让较轻的检测器超过更复杂的逐帧估计。
- [BroadTrack](https://openaccess.thecvf.com/content/WACV2025/html/Magera_BroadTrack_Broadcast_Camera_Tracking_for_Soccer_WACV_2025_paper.html) 表明镜头畸变、光流和云台物理约束对长视频稳定标定有效；其畸变消融是单项提升最大的部分。
- [AuxFlow](https://www.sciencedirect.com/science/article/pii/S1077314226000299) 进一步验证了“可靠锚帧 + 光流辅助对应点”对视频球场注册和平面球员定位的价值。
- [TacticalCalib](https://openaccess.thecvf.com/content/WACV2026/html/Fan_TacticalCalib_End-to-End_6-DoF_Camera_Pose_Regression_for_Tactical_Camera_Calibration_WACV_2026_paper.html) 是值得测试的单帧初始化对照，但本项目已知固定机位与水平转动约束，直接估计完整 6-DoF 属于过参数化。

## 4. 推荐架构：Constrained Pitch Tracker

### 4.1 静态相机模型

每个摄像头保存一个 `CameraRigProfile`：

```python
@dataclass(frozen=True)
class CameraRigProfile:
    camera_id: str
    image_size: tuple[int, int]
    lens_model: Literal["pinhole", "fisheye"]
    camera_matrix: np.ndarray
    distortion: np.ndarray
    pitch_length_m: float
    pitch_width_m: float
    pan_axis_origin_xyz_m: np.ndarray
    pan_axis_direction: np.ndarray
    optical_center_offset_m: np.ndarray
    base_rotation: np.ndarray
    fixed_tilt_rad: float
    fixed_roll_rad: float
    fixed_focal_px: float
    pan_zero_rad: float
```

比赛中默认冻结：

- 相机内参；
- 畸变系数；
- 焦距；
- 云台旋转轴、相机高度和安装位置；
- 光心相对旋转轴的固定偏移；
- 俯仰和横滚；
- 场地尺寸。

每帧动态状态只保留：

```text
x_t = [pan_t, pan_velocity_t]
```

若实测发现相机存在小幅俯仰漂移，可扩展成 `[pan, pan_velocity, tilt_bias]`，但不能一开始就恢复成无约束 8-DoF。

### 4.2 镜头模型选择

同一组棋盘格/Charuco 标定图分别拟合：

1. OpenCV pinhole + radial/tangential distortion；
2. OpenCV fisheye 四系数模型。

按以下验证集指标选择，不按镜头名称猜测：

- 标定图重投影误差；
- 图像边缘直线残差；
- 球场标线在整幅画面上的重投影误差；
- undistort 后有效视野比例。

[OpenCV fisheye 官方模型](https://docs.opencv.org/4.13.0/db/d58/group__calib3d__fisheye.html) 支持独立的标定、点去畸变和预计算 remap。运行时继续使用预计算映射，成本只是一遍 `remap`。

### 4.3 球场感知模型

采用一个共享 backbone、两个高分辨率 head：

```text
RGB 512×288 或 640×360
  ↓
轻量高分辨率 backbone
  ├── Semantic Segment Head：每条直线/圆弧的语义概率图
  └── Landmark Heatmap Head：真实交点、线圆交点、罚球点等热图
```

不使用 YOLO 实例分割作为主候选，因为球场白线细且长，低分辨率实例 mask 容易断裂。第一轮必须同数据训练并比较：

| 候选 | 设计目的 | 预期取舍 |
|---|---|---|
| MobileNetV3-Large + LR-ASPP | 最低算力基线 | 快，但细线边界可能弱 |
| PIDNet-S | 主候选 | 高分辨率 detail/boundary 分支适合细线与交点 |
| SegFormer-B0 | 鲁棒性对照 | 全局上下文更强，但 TensorRT 和延迟需实测 |

[MobileNetV3/LR-ASPP](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html)、[PIDNet](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html) 和 [SegFormer](https://proceedings.neurips.cc/paper_files/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html) 都有公开论文与实现。最终选择必须由球场数据上的线定位误差和 RTX 5060 Ti 延迟共同决定，不能套用 Cityscapes mIoU 排名。

#### 输出语义

线段必须带身份，而不只是 `white_line`：

- 左/右球门线；
- 上/下边线；
- 中线；
- 左/右禁区与小禁区各边；
- 中圈、罚球弧；
- 罚球点、中点；
- 可选球门柱等非地面元素，仅用于 3D 校验。

圆弧与直线应分开建模。训练标签可同时保留“元素实例”和“几何族”，以便在局部画面中进行语义消歧。

### 4.4 单帧初始化与重定位

初始化采用四级证据：

1. 网络直接输出的高置信度关键点；
2. 从语义线拟合得到的线端点和线线交点；
3. 线与圆锥曲线的交点、圆弧采样点；
4. 已知相机安装位置和允许 pan 范围产生的先验。

处理步骤：

```text
亚像素热图解码
  ↓
按置信度排序的候选对应
  ↓
MAGSAC++ 初始单应性/姿态
  ↓
退化检查：点数、方向族、凸包覆盖、条件数
  ↓
固定 K/D/C/focal/tilt/roll，仅优化 pan（初始化时可放开少量静态参数）
  ↓
点到点 + 点到语义线/圆弧的鲁棒非线性最小二乘
  ↓
双向投影验证与 CameraState
```

用 `USAC_MAGSAC` 替换 `RANSAC → RHO → LMEDS` 的串行试错。MAGSAC++ 通过对噪声尺度边缘化和迭代重加权提高几何估计精度，并在公开单应性数据上降低失败率；见 [MAGSAC++ 论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.html)。

非线性目标：

```text
E = w_p · robust(point reprojection error)
  + w_l · robust(segment distance-transform error)
  + w_c · robust(circle/conic error)
  + w_prior · camera prior error
```

各模态先按验证集噪声尺度归一化，再确定权重，不能直接混合像素距离和归一化距离。

### 4.5 逐帧云台跟踪

普通帧不运行完整重定位：

1. 在上一帧已确认的球场线、草坪纹理和静态背景中均匀采样点；
2. 排除球员框、裁判框、广告屏动态区域和画面上部观众区域；
3. 使用 pyramidal Lucas–Kanade 稀疏光流；
4. 前后向误差和局部 RANSAC 过滤错误匹配；
5. 用观测更新 `[pan, pan_velocity]` EKF；
6. 从物理相机模型重新生成当前帧投影矩阵。

这样得到的 `H_t` 是相机状态的派生量，不再直接对 9 个矩阵元素做普通线性平滑。

#### 自适应关键帧调度

```text
高置信度、低转速：每 10 帧运行感知模型
正常跟踪：每 5 帧
快速转动或置信度下降：下一帧立即运行
硬切镜头/光流崩溃：清空状态并重定位
连续重定位失败：LOST，不输出伪坐标
```

光流不能无限传播。每次语义检测都用球场线残差纠偏，防止积分漂移。

### 4.6 状态与置信度

替换仅有 `FRESH/REUSED/STALE/UNAVAILABLE` 的粗粒度状态：

```python
class CameraTrackingStatus(str, Enum):
    INITIALIZING = "initializing"
    RELOCALIZED = "relocalized"   # 当前帧完成全局重定位
    CORRECTED = "corrected"       # 当前帧有语义模型校正
    TRACKED = "tracked"           # 光流 + 滤波更新
    PREDICTED = "predicted"       # 无可靠观测，仅短时预测
    LOST = "lost"
```

每帧输出：

```python
@dataclass
class CameraState:
    status: CameraTrackingStatus
    pan_rad: float
    pan_velocity_rad_s: float
    covariance: np.ndarray
    image_to_pitch: np.ndarray | None
    pitch_to_image: np.ndarray | None
    point_inliers: int
    visible_segment_count: int
    spatial_coverage: float
    mean_segment_error_px: float | None
    p95_segment_error_px: float | None
    confidence: float
    age_since_semantic_update: int
```

置信度至少由以下量组成：

- 点/线观测数量；
- 两个以上非平行方向族；
- 图像覆盖面积；
- 点、线重投影误差；
- 光流前后向误差和有效比例；
- EKF innovation；
- pan 加速度是否物理合理；
- 当前矩阵与上一矩阵在场地网格上的投影跳变量。

`PREDICTED` 可以用于画面连续显示，但越位、犯规定位和距离统计只接受 `RELOCALIZED/CORRECTED/TRACKED` 且置信度达到阈值的状态。

### 4.7 球员地面接触点

推荐新增一个只预测 **单个地面接触点** 的轻量 head，而不是为所有人运行完整 17 点人体姿态：

```text
person box / shared feature
  ↓
1-keypoint contact head
  ↓
left/right foot 可见时取双脚支撑中心
遮挡时输出接触点概率分布和不确定度
```

运行策略：

1. 高置信度 contact point；
2. 可选轻量 ankle pose 的双踝中点；
3. 完整检测框的底部中心；
4. 检测框截断或严重遮挡时不投影，而不是强行给坐标。

SoccerNet Game State Reconstruction 数据包含逐帧球员场地位置和球场标线/相机信息，可从真值相机投影反推出图像中的接触点，用于预训练；官方任务与数据说明见 [SoccerNet GSR](https://arxiv.org/abs/2404.11335)。之后使用本机位少量人工修正帧微调。

投影结果携带：

```python
@dataclass
class PitchCoordinate:
    xy_m: tuple[float, float] | None
    sigma_m: float | None
    source: Literal["contact_head", "ankles", "bbox_bottom", "none"]
    camera_status: CameraTrackingStatus
```

球员坐标误差由“图像接触点误差”和“相机状态协方差”共同传播。球场外点、矩阵奇异点、误差上界过大的点输出 `None`。

> 足球不总在地面。单应性只能得到足球在地面平面的投影；空中球不能被当作真实 2D 地面位置。第一版应输出 `ground_projection` 和不确定度，空中高度需要单目轨迹模型或多摄像头才能进一步解决。

## 5. 数据与训练方案

### 5.1 公共数据

1. **SoccerNet Calibration**：20,028 张来自 500 场比赛的图像，带球场元素标注，适合点/线语义检测和单帧标定。
2. **SoccerNet GSR**：200 个 30 秒视频序列，包含逐帧球场线点和球员场地坐标，适合时序标定与最终球员定位评估。
3. **TS-WorldCup/CARWC**：用于关键点时序和跨比赛泛化对照。

数据来源与评测协议以 [SoccerNet Camera Calibration](https://www.soccer-net.org/tasks/camera-calibration) 和 [SoccerNet sn-calibration](https://github.com/SoccerNet/sn-calibration) 为准。

### 5.2 自有机位数据

必须增加自有摄像头域数据，因为镜头畸变、安装高度、草坪色彩和画质与转播数据不同。

建议采集：

- 白天、阴影、夜间灯光；
- 慢速/快速左右 pan；
- 左端、中央、右端视角；
- 球员稀少和密集遮挡；
- 雨天或低对比度条件；
- 每种条件至少一段不参与训练的连续测试序列。

标注策略：

1. 人工精确对齐少量锚帧的球场模板；
2. 利用受约束相机模型和光流传播到相邻帧；
3. 人工复核高残差帧；
4. 从投影模板自动生成每条线、圆弧和关键点标签；
5. 按完整视频序列切分 train/validation/test，禁止相邻帧跨集合泄漏。

### 5.3 损失函数

```text
L = λ_seg · focal/dice semantic segment loss
  + λ_boundary · boundary loss
  + λ_kp · heatmap focal loss
  + λ_offset · subpixel offset loss
  + λ_geo · differentiable geometry consistency loss
```

类别采样要补偿细线像素占比小的问题；强增强不能破坏几何标签。允许亮度、色彩、模糊、压缩、遮挡增强，不允许未同步更新标注的随机透视变换。

## 6. 软件模块边界

建议新建独立包，不在旧 `PitchProjectionEngine` 内继续堆条件：

```text
app/field_registration/
├── types.py                 # observation/state/result contracts
├── pitch_model.py           # metric field geometry and venue profiles
├── lens.py                  # pinhole/fisheye models and remap cache
├── perception.py            # segment/keypoint backend interface
├── decode.py                # heatmap/segment subpixel decoding
├── initializer.py           # MAGSAC++ and degeneracy checks
├── point_line_refiner.py    # robust constrained optimization
├── optical_flow.py          # masked LK and forward-backward checks
├── pan_filter.py            # 1D EKF
├── tracker.py               # state machine and scheduling
├── confidence.py            # quality score and safety gates
├── contact_point.py         # player ground-contact point
├── projection.py            # image↔pitch and uncertainty propagation
└── metrics.py               # JaC, errors, jitter, recovery, latency
```

稳定对外接口：

```python
class FieldRegistrationCore:
    def process(
        self,
        frame: np.ndarray,
        frame_index: int,
        person_masks_or_boxes: object | None = None,
    ) -> FieldRegistrationFrame:
        ...
```

旧 `VisionCore` 只消费 `FieldRegistrationFrame`，不再负责关键点调度和矩阵生命周期。

## 7. 分阶段实施计划

### 阶段 A：真值与当前基线

交付：

- 从 `test1.mp4`、`test2.mp4` 和真实相机视频各选困难片段；
- 人工标注球场线/关键点锚帧和至少 200 个球员接触点；
- 建立 JaC、线重投影误差、网格投影误差、球员米制误差和时序抖动评测；
- 重新测量当前 32 点方案，不再只记录 available ratio。

完成条件：能明确回答“误差来自镜头、球场检测、矩阵估计还是脚点”。

### 阶段 B：相机与场地物理模型

交付：

- pinhole/fisheye 标定对比；
- 每场馆真实尺寸 profile；
- `CameraRigProfile` 持久化和校验；
- 多个 pan 锚帧联合估计相机位置、固定 tilt/roll 和 pan zero；
- image↔pitch 投影及数值 Jacobian/协方差传播测试。

完成条件：人工真值点投影时，边缘残差不再呈明显径向规律。

### 阶段 C：球场感知模型竞赛

在同一数据、输入分辨率和增强配置下训练：

- C0：当前 32 点模型；
- C1：MobileNetV3-LR-ASPP 双 head；
- C2：PIDNet-S 双 head；
- C3：SegFormer-B0 双 head；
- C4：PnLCalib 官方权重，仅作为精度上界/速度参考。

完成条件：根据关键点 PCK、语义线距离、遮挡分组结果和 TensorRT 延迟选择一个模型。

### 阶段 D：单帧初始化

交付：

- 亚像素关键点；
- 线/圆弧拟合与虚拟交点；
- USAC_MAGSAC；
- 退化检测；
- 点线联合受约束 refinement；
- 双向验证和失败状态。

完成条件：不能通过质量门禁的帧必须返回失败，不允许输出视觉上合理但错误的矩阵。

### 阶段 E：时序 pan tracker

交付：

- 动态区域 mask；
- 均匀采样 LK 光流；
- 前后向检查；
- 1D pan EKF；
- 自适应关键帧调度；
- 镜头硬切/快速 pan 检测；
- 置信度下降后的自动重定位。

完成条件：左右快速 pan 和短时标线遮挡期间不复用错误矩阵，恢复后无明显跳变。

### 阶段 F：地面接触点与投影

交付：

- 单接触点模型或 head；
- bbox/ankle 回退策略；
- 图像边缘与遮挡门禁；
- 米制坐标误差和不确定度；
- 球员、裁判、守门员统一接触点接口。

完成条件：相对于 bbox bottom-center，测试集球员场地坐标误差和速度尖峰显著下降。

### 阶段 G：CUDA、TensorRT 与长视频验收

交付：

- FP16 ONNX/TensorRT；
- 30–60 分钟连续运行；
- 与球员检测、跟踪、分队同时运行；
- 失败片段视频、误差曲线和性能报告；
- 完成消融并确定生产配置。

## 8. 消融实验矩阵

### 8.1 几何与时序

| 实验 | 关键点 | 语义线/圆 | 物理约束 | 光流 | EKF | 接触点 |
|---|---:|---:|---:|---:|---:|---:|
| P0 当前基线 | 32 | 否 | 否 | 否 | 否 | bbox |
| P1 | 新模型 | 否 | 否 | 否 | 否 | bbox |
| P2 | 新模型 | 是 | 否 | 否 | 否 | bbox |
| P3 | 新模型 | 是 | pan-only | 否 | 否 | bbox |
| P4 | 新模型 | 是 | pan-only | 是 | 否 | bbox |
| P5 推荐几何 | 新模型 | 是 | pan-only | 是 | 是 | bbox |
| P6 完整方案 | 新模型 | 是 | pan-only | 是 | 是 | contact |

### 8.2 镜头

| 实验 | 模型 | 目的 |
|---|---|---|
| L0 | 不去畸变 | 量化畸变影响 |
| L1 | pinhole radial/tangential | 当前类别模型 |
| L2 | fisheye | 广角候选 |
| L3 | 在优化中保留一个小 distortion bias | 检查温漂/标定残差，默认关闭 |

### 8.3 调度

```text
固定 interval = 1 / 3 / 5 / 10
自适应 interval = 5↔10，快速 pan/低置信度立即刷新
光流 = 无 / LK / CUDA optical flow（只在 CPU 成本过高时考虑）
```

## 9. 指标与验收标准

### 9.1 球场感知

- 关键点 PCK@5px / PCK@10px；
- 每类语义线 precision、recall、Chamfer/点到线距离；
- 遮挡、阴影、边缘、快速 pan 分组指标；
- 有效几何方向族数量与图像覆盖率。

### 9.2 标定与投影

- SoccerNet JaC@5 / JaC@10；
- mean / median / P95 segment reprojection error；
- 可见区域与全场 IoU；
- 规则网格 image→pitch 投影误差（米）；
- completeness；
- 错误矩阵通过门禁率，目标为 0；
- 重定位平均/P95 帧数。

### 9.3 时序稳定性

- pan 速度和加速度尖峰；
- 相邻帧球场模板抖动；
- 静态测试点的场地坐标标准差；
- 每分钟 LOST 次数；
- 快速 pan、遮挡和硬切后的恢复时间；
- `TRACKED/PREDICTED/LOST` 占比。

### 9.4 下游坐标

- 球员接触点图像误差；
- 球员场地坐标 median/P90/P95 米制误差；
- 坐标缺失率；
- 不合理速度尖峰次数；
- 按画面上/中/下区域分组的误差。

### 9.5 性能

RTX 5060 Ti 初始工程目标：

- 语义/关键点模型单次 FP16 延迟 ≤ 15 ms；
- 常规每 5 帧运行时，标定链路平均摊销 ≤ 4 ms/frame；
- 光流 + pan filter P95 ≤ 2 ms/frame；
- 相比关闭球场注册的人员主链路，完整 FPS 下降 ≤ 10%；
- 新增显存 ≤ 500 MB；
- 连续 60 分钟无持续显存增长。

精度目标应先通过阶段 A 建立基线再冻结。可以先使用以下候选门槛：

- 支持的主机位片段 completeness ≥ 99.5%；
- 相比 P0，P95 线重投影误差下降 ≥ 40%；
- 相比 P0，快速 pan 片段的坐标跳变下降 ≥ 50%；
- 人工标注球员坐标 median error ≤ 0.75 m、P90 ≤ 1.5 m；
- 任何质量门禁失败帧不得输出“可用于越位”的可信坐标。

这些是工程目标，不是当前已达到的结果。

## 10. 明确不采用的主线

- **不把完整 PnLCalib 每帧运行**：点线思想正确，但官方双 HRNet-W48 和多配置 voting 过重。
- **不直接回归任意 homography/6-DoF 作为唯一结果**：域外画面难以自验证，也没有利用固定机位约束。
- **不只用 Hough 白线**：无法可靠区分线的语义身份，观众席、广告牌和球衣会产生干扰。
- **不直接平滑 9 个 homography 元素**：矩阵存在尺度自由度，普通 EMA 可能产生非物理解。
- **不把旧矩阵长时间当有效结果**：无观测时只允许短期 `PREDICTED`，并携带增长的不确定度。
- **不为地面接触点运行完整人体姿态作为默认路径**：17 点姿态对本任务冗余；优先单接触点 head。
- **不要求云台硬件角度才能工作**：视频本身足够估计 pan；若未来能读取编码器角度，可作为额外观测而不是系统前提。

## 11. 最终推荐生产配置候选

```text
离线：pinhole/fisheye 二选一 + 多 pan 锚帧联合安装标定

在线关键帧：
PIDNet-S 双 head（待 C1/C2/C3 实测确认）
+ 亚像素关键点
+ 语义线/圆弧
+ USAC_MAGSAC 初始化
+ 受约束点线 refinement

在线普通帧：
masked pyramidal LK
+ [pan, pan_velocity] EKF
+ 物理相机模型派生 homography
+ 自适应 5/10 帧语义刷新
+ 低置信度立即重定位

玩家投影：
单地面接触点 head
+ bbox bottom-center 安全回退
+ 坐标不确定度
+ 下游可信状态门禁
```

实施时首先完成阶段 A 和 B。没有真实误差基线与正确镜头/场地模型之前，不应先训练更大的关键点网络，因为那会把几何、畸变和接触点误差混在一起，无法判断优化是否真正有效。
