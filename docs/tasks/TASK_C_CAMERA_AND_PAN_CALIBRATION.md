# 任务 C：真实相机内参与水平云台标定

## 0. 你领取的任务是什么

你负责真实硬件，不负责训练模型。目标是回答两个问题：

1. 这台相机和镜头怎样把广角画面可靠地去畸变；
2. 相机安装到固定云台后，只左右旋转时，怎样把每个角度映射到球场坐标。

最终交付两个核心文件：

```text
camera.npz       镜头内参和畸变参数
camera-rig.json  固定安装位置和水平 pan 几何模型
```

你需要有：真实相机、实际使用的镜头、最终云台/支架、棋盘格、能看到足球场的安装位置。采集和标定都只需要 CPU，不需要 NVIDIA GPU。

预计用时：环境 20–40 分钟，棋盘格采集 30–60 分钟，现场 pan 采集 30–60 分钟，标注和复核 60–120 分钟。

## 1. 开始前必须确认的硬件条件

- [ ] 使用最终比赛要用的同一台相机和同一支镜头；
- [ ] 能固定分辨率，例如 1920×1080；
- [ ] 能固定焦距/变焦档位，标定后不再变焦；
- [ ] 尽量关闭会改变内参的数字防抖、自动裁切和电子变焦；
- [ ] 云台能只做左右 pan，采集时不改变 tilt 和 roll；
- [ ] 有硬质平整棋盘格；
- [ ] 能获得真实球场长宽，或负责人明确允许暂用 105×68 m；
- [ ] 原始图片和视频有可靠存储位置，不经过聊天软件压缩。

如果最终焦距、分辨率或数字防抖还没决定，先不要拍棋盘格。设置改变后必须重新标定。

## 2. 安装 Git 和 uv

检查 Git：

```bash
git --version
```

没有 Git 时从 [Git 官网](https://git-scm.com/downloads) 安装。

安装 uv。

macOS/Linux：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

重新打开终端：

```bash
uv --version
```

本任务使用轻量 `uv run --no-project` 命令，只安装 NumPy 和 OpenCV。

## 3. 从 GitHub 创建硬件任务分支

先决定唯一 `camera_id`，建议格式：

```text
场地缩写-机位-相机型号
```

例如：`school-main-sony-zve10`。只使用小写字母、数字和短横线。

克隆代码并创建分支，把 `<camera-id>` 换成真实值：

```bash
git clone https://github.com/refereelink-team/refereelink-backend.git
cd SC
git fetch origin
git switch --create data/camera-<camera-id> origin/codex/dev-pitch-projection
git branch --show-current
```

如果远端不存在 `codex/dev-pitch-projection`，停止并联系负责人，不要从 `main` 开始。

## 4. 建立本地目录

以下以 `school-main-sony-zve10` 为例：

```bash
mkdir -p task_data/camera-school-main-sony-zve10/chessboard
mkdir -p task_data/camera-school-main-sony-zve10/lens-check
mkdir -p task_data/camera-school-main-sony-zve10/rig
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force task_data\camera-school-main-sony-zve10\chessboard
New-Item -ItemType Directory -Force task_data\camera-school-main-sony-zve10\lens-check
New-Item -ItemType Directory -Force task_data\camera-school-main-sony-zve10\rig
```

原始数据结构最终应类似：

```text
task_data/camera-school-main-sony-zve10/
├── camera-settings.json
├── chessboard/                  30–50 张原图
├── camera.npz
├── camera.report.json
├── lens-check/                  原图/去畸变对比图
└── rig/
    ├── pan-anchors.mp4
    ├── pack/
    │   ├── frames/
    │   ├── manifest.json
    │   └── manifest.annotated.json
    ├── validation-report.json
    ├── camera-rig.json
    └── camera-rig.report.json
```

`task_data/` 是本机工作区，不提交 Git。

## 5. 记录相机设置

创建 `camera-settings.json`：

```json
{
  "camera_id": "school-main-sony-zve10",
  "camera_model": "填写型号",
  "lens_model": "填写镜头型号",
  "resolution": [1920, 1080],
  "fps": 30,
  "zoom_or_focal_length": "固定数值或档位",
  "focus_mode": "manual 或 autofocus",
  "digital_stabilization": false,
  "electronic_crop": false,
  "mount_height_m": null,
  "pitch_length_m": 105.0,
  "pitch_width_m": 68.0,
  "chessboard_inner_corners": [9, 6],
  "chessboard_square_size_mm": 25.0,
  "capture_date": "YYYY-MM-DD",
  "notes": "天气、安装位置、无法确认的设置"
}
```

不知道的字段填 `null` 或写明未知，不要猜。特别确认图片和 pan 视频的分辨率、宽高比、焦距和防抖设置完全一致。

## 6. 准备棋盘格

默认工具识别 `9×6` 个内角点，因此打印板必须有：

```text
10×7 个黑白方格
9×6 个内部交点
```

要求：

- 打印后贴在平整硬板上，不能卷曲；
- 用直尺或卡尺测量单个方格边长；
- `--square-size` 使用实测值，例如 25 mm；
- 黑白边界清晰，不要使用屏幕显示的棋盘格做正式标定；
- 棋盘必须完整进入画面，所有 9×6 内角都可见。

## 7. 拍摄 30–50 张棋盘格照片

固定最终分辨率、焦距、防抖和对焦策略后开始拍摄。

必须覆盖：

- 画面正中央；
- 左上、右上、左下、右下；
- 靠近四条边缘；
- 近、中、远三种距离；
- 正视、水平倾斜、垂直倾斜和轻微旋转；
- 棋盘占画面约 20% 到 70% 的不同大小。

不要连续站在一个位置只拍几十张几乎相同的照片。每张拍摄前都改变棋盘位置、距离或姿态。

删除以下图片：

- 运动模糊或失焦；
- 过曝导致白格边缘消失；
- 棋盘被裁切；
- 棋盘弯曲；
- 分辨率与其他图片不同；
- 自动变焦或电子裁切发生变化。

将保留的原图放入：

```text
task_data/camera-school-main-sony-zve10/chessboard/
```

建议命名 `image-0001.jpg`、`image-0002.jpg`，避免中文、空格和重复文件名。

## 8. 运行 pinhole/fisheye 自动比较

在仓库根目录执行；把路径和方格尺寸替换为实际值：

```bash
uv run --no-project \
  --with "numpy>=1.26" \
  --with "opencv-python>=4.8" \
  python -m tools.calibrate_camera \
  --images task_data/camera-school-main-sony-zve10/chessboard \
  --output task_data/camera-school-main-sony-zve10/camera.npz \
  --report task_data/camera-school-main-sony-zve10/camera.report.json \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-size 25 \
  --lens-model auto
```

成功时终端会打印两个候选模型的训练 RMS 和留出误差，并显示：

```text
Saved .../camera.npz
Saved .../camera.report.json
Selected lens model: pinhole 或 fisheye
```

打开 `camera.report.json`，记录：

- `source_image_count`；
- `valid_image_count`；
- `selected_model`；
- 两个 candidate 的 `validation_median_px`、`validation_p95_px`；
- `failures`。

初始人工目标：

```text
valid_image_count >= 20
所选模型 validation_median_px 尽量 < 1.0 px
所选模型 validation_p95_px 尽量 < 2.0 px
```

这些是建议目标，不是为了通过而修改报告的理由。若有效图少于 20 或误差明显过大，补拍更多边缘、角落和不同倾斜角度的清晰照片后重新运行。

常见失败：

| 提示 | 原因与处理 |
|---|---|
| `No calibration images were found` | 路径错误或图片扩展名不支持 |
| `At least 3 valid...` | 内角识别失败；检查 10×7 方格、清晰度和完整性 |
| `All calibration images must have the same size` | 混入不同分辨率图片，移出后重跑 |
| `auto requires at least six valid...` | 有效图片不足，继续采集 |
| fisheye `calibration failed` | 数据覆盖或数值条件不足；补拍边缘和倾斜视角，不要直接伪造通过 |

## 9. 生成人工去畸变检查图

执行：

```bash
uv run --no-project \
  --with "numpy>=1.26" \
  --with "opencv-python>=4.8" \
  python -m tools.render_camera_calibration_check \
  --images task_data/camera-school-main-sony-zve10/chessboard \
  --calibration task_data/camera-school-main-sony-zve10/camera.npz \
  --output task_data/camera-school-main-sony-zve10/lens-check \
  --max-images 8
```

每张输出图左侧是 `SOURCE`，右侧是 `RECTIFIED`。逐张检查：

- 画面边缘原本弯曲的直线是否明显变直；
- 右图有没有异常波浪、局部拉伸或折叠；
- 四角裁切是否可以接受；
- 棋盘直线在右图是否连续；
- 所有输出尺寸是否一致。

不能只凭 `camera.report.json` 的数字宣布完成。至少把 8 张对比图交给负责人复核。

## 10. 在最终安装位置采集水平 pan 视频

把相机安装到比赛时的最终位置。此后：

- 支架基座不能移动；
- 相机高度不能变；
- 焦距、分辨率、防抖不能变；
- tilt 和 roll 固定；
- 只允许水平左右 pan。

录制建议：

1. 从最左可用角度开始，静止 3 秒；
2. 转到偏左，静止 3 秒；
3. 转到中左，静止 3 秒；
4. 转到中央，静止 3 秒；
5. 转到中右、偏右、最右，各静止 3 秒；
6. 推荐总共 7–9 个不同锚角度；
7. 转动过程可以录下，但后面只标静止且清晰的帧；
8. 每个角度画面中尽量能看到 8 个以上球场白线交点。

保存原始文件：

```text
task_data/camera-school-main-sony-zve10/rig/pan-anchors.mp4
```

如果有云台编码器，另外记录每个停留角度的度数和对应视频时间。没有编码器也可以完成任务，禁止目测伪造角度。

## 11. 从 pan 视频生成锚帧标注包

用 30 帧覆盖整段视频，之后只标其中清晰、静止且角度互不重复的 7–9 帧：

```bash
uv run --no-project \
  --with "numpy>=1.26" \
  --with "opencv-python>=4.8" \
  python -m tools.create_pitch_annotation_pack \
  task_data/camera-school-main-sony-zve10/rig/pan-anchors.mp4 \
  task_data/camera-school-main-sony-zve10/rig/pack \
  --frames 30 \
  --split calibration \
  --pitch-length-m 105 \
  --pitch-width-m 68
```

检查空包：

```bash
uv run --no-project \
  --with "numpy>=1.26" \
  --with "opencv-python>=4.8" \
  python -m tools.validate_pitch_annotations \
  task_data/camera-school-main-sony-zve10/rig/pack/manifest.json \
  --allow-empty
```

## 12. 标注 7–9 个 pan 锚帧

启动页面：

```bash
uv run --no-project python -m http.server 8765 \
  --bind 127.0.0.1 \
  --directory task_data/camera-school-main-sony-zve10/rig/pack
```

浏览器打开：

```text
http://127.0.0.1:8765/index.html
```

本任务只需要 `Pitch landmarks`，不需要标球员接触点。

选择 7–9 个满足以下条件的帧：

- 相机处于静止状态，没有运动模糊；
- 覆盖从最左到最右；
- 相邻锚帧不是同一个 pan 角度；
- 每帧至少 4 个可靠锚点，推荐 8–12 个；
- 锚点覆盖两个不平行方向和较大画面区域。

坐标约定：

```text
远端边线 = top = Y=0
左球门 = left = X=0
右球门 = right = X=球场长度
相机侧边线 = bottom = Y=球场宽度
```

云台转向后约定仍不改变。只标实际可见交点，不外推、不猜测。

其他未选中的帧保持完全空白，不要为了让时间轴变绿随便加点。

完成后点击 `Export JSON`，复制到：

```text
task_data/camera-school-main-sony-zve10/rig/pack/manifest.annotated.json
```

如果有准确编码器数据，先把“视频时间—角度”表交给负责人确认，再决定是否向对应 frame 添加 `encoder_pan_rad`。没有编码器时不要编辑这个字段。

## 13. 校验 pan 锚点

```bash
uv run --no-project \
  --with "numpy>=1.26" \
  --with "opencv-python>=4.8" \
  python -m tools.validate_pitch_annotations \
  task_data/camera-school-main-sony-zve10/rig/pack/manifest.annotated.json \
  --output task_data/camera-school-main-sony-zve10/rig/validation-report.json \
  --allow-no-contact-points
```

rig 包不要求球员接触点，`--allow-no-contact-points` 会只验证几何标注。必须满足：

- `valid=true`、`issues=[]`；
- 所有有锚点帧的 `frames[i].issues=[]`；
- 每帧 `correspondence_count >= 4`，推荐 8–12；
- `homography_available=true`；
- `reprojection_median_px` 尽量小于 3 px；
- 没有 `missing_frame_image`、几何退化和覆盖不足。

## 14. 生成 CameraRigProfile

根据云台真实左右极限替换 `--pan-min-deg` 和 `--pan-max-deg`。不知道时可以先用保守范围，但必须在复核记录中注明是假设值。

```bash
uv run --no-project \
  --with "numpy>=1.26" \
  --with "opencv-python>=4.8" \
  python -m tools.calibrate_camera_rig \
  --annotations task_data/camera-school-main-sony-zve10/rig/pack/manifest.annotated.json \
  --lens task_data/camera-school-main-sony-zve10/camera.npz \
  --camera-id school-main-sony-zve10 \
  --output task_data/camera-school-main-sony-zve10/rig/camera-rig.json \
  --report task_data/camera-school-main-sony-zve10/rig/camera-rig.report.json \
  --pan-min-deg -90 \
  --pan-max-deg 90
```

成功标志：

```text
Saved .../camera-rig.json
Saved .../camera-rig.report.json
```

报告必须满足：

```text
accepted = true
issues = []
camera_centre_spread_m <= 0.5
maximum_anchor_reprojection_error_px <= 3.0
```

失败时不要增大门槛绕过。按顺序检查：

1. 是否使用了同一焦距和分辨率的 `camera.npz`；
2. 球场长宽是否正确；
3. `left/right/top/bottom` 是否标反；
4. 是否标了相机运动中的模糊帧；
5. 支架是否移动或 pan 时伴随 tilt/roll；
6. 某个锚帧是否有明显错误点。

修正数据后重新生成，直到通过或把真实硬件问题报告给负责人。

## 15. 准备 GitHub 交付物

原始照片、原始视频和标注包 JPG 不提交 Git。创建目录：

```bash
mkdir -p submissions/pitch-registration/cameras/school-main-sony-zve10
```

复制小型结果文件：

```bash
cp task_data/camera-school-main-sony-zve10/camera-settings.json \
  submissions/pitch-registration/cameras/school-main-sony-zve10/
cp task_data/camera-school-main-sony-zve10/camera.npz \
  submissions/pitch-registration/cameras/school-main-sony-zve10/
cp task_data/camera-school-main-sony-zve10/camera.report.json \
  submissions/pitch-registration/cameras/school-main-sony-zve10/
cp task_data/camera-school-main-sony-zve10/rig/pack/manifest.annotated.json \
  submissions/pitch-registration/cameras/school-main-sony-zve10/rig-manifest.annotated.json
cp task_data/camera-school-main-sony-zve10/rig/camera-rig.json \
  submissions/pitch-registration/cameras/school-main-sony-zve10/
cp task_data/camera-school-main-sony-zve10/rig/camera-rig.report.json \
  submissions/pitch-registration/cameras/school-main-sony-zve10/
```

Windows PowerShell 使用 `New-Item -ItemType Directory -Force` 创建同名目录，并用 `Copy-Item` 复制上述六个文件；目标文件名保持完全一致。

创建 `review-notes.md`：

```markdown
# 相机与云台标定交付

- 负责人：
- camera_id：
- 相机/镜头：
- 分辨率/FPS/焦距：
- 棋盘方格实测尺寸：
- 原始棋盘图片数量：
- 有效图片数量：
- selected_model：
- validation median/P95：
- 去畸变对比图人工结论：
- 相机最终安装位置：
- 球场实际尺寸及来源：
- pan 锚帧数量：
- 是否有编码器角度：
- rig accepted：
- camera centre spread：
- maximum anchor error：
- 原始照片、视频、lens-check 和完整 pack 存放位置：
- 已知限制或假设：
```

对原始照片目录、pan 视频、`camera.npz` 和 `camera-rig.json` 分别计算 SHA-256；目录可先压缩成 ZIP 再计算哈希。把哈希写入复核记录。

提交：

```bash
git add submissions/pitch-registration/cameras/school-main-sony-zve10
git commit -m "data(field): calibrate school main camera rig"
git push -u origin HEAD
```

创建 GitHub PR：

- base：`codex/dev-pitch-projection`；
- compare：你的 `data/camera-<camera-id>`；
- 标题：`data(field): calibrate <camera-id> rig`；
- PR 附原始数据交接链接和哈希；
- 不提交 MP4 和几十张原始照片。

## 16. 任务完成清单

### 镜头部分

- [ ] 相机设置和棋盘尺寸已记录；
- [ ] 30–50 张不同位置、距离和倾角的棋盘格照片；
- [ ] 推荐至少 20 张有效图片；
- [ ] pinhole/fisheye auto 已运行；
- [ ] `camera.npz` 和 `camera.report.json` 已生成；
- [ ] 8 张去畸变对比图已人工复核。

### 云台部分

- [ ] 相机位于最终安装位置；
- [ ] 焦距、分辨率、tilt、roll 固定；
- [ ] 录制 7–9 个从左到右的静止锚角度；
- [ ] 每个锚帧推荐 8–12 个可靠球场点；
- [ ] `camera-rig.report.json` 中 `accepted=true`、`issues=[]`；
- [ ] 原始数据、报告和哈希已交接。

### GitHub

- [ ] PR base 是 `codex/dev-pitch-projection`；
- [ ] PR 包含配置、NPZ、JSON、报告和复核记录；
- [ ] PR 不包含原始视频和大批照片；
- [ ] 所有假设和不确定参数均明确写出，没有猜测。

全部勾选后，任务 C 才算完成。CUDA 长跑由主开发者在合并后执行，不属于本任务。
