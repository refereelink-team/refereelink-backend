# 固定水平云台相机标定

运行时物理状态只估计水平 pan。相机内参、畸变、光心位置、基础姿态、焦距和球场尺寸必须在部署前冻结。

## 1. 采集棋盘格

- 同一分辨率和固定焦距；
- 棋盘覆盖画面中心、四角和边缘；
- 改变棋盘距离、俯仰和旋转，避免所有视图近似相同；
- `auto` 最少需要 6 张成功检测的图片，建议 15–30 张；
- 留出若干不参与拟合的球场白线画面，用于检查边缘直线残差。

## 2. pinhole/fisheye 留出验证

```bash
uv run python tools/calibrate_camera.py \
  --images /path/to/chessboard-images \
  --output assets/calibration/camera.npz \
  --lens-model auto \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-size 0.025 \
  --validation-folds 8
```

`auto` 对每个候选模型执行按视图留出验证，选择分数为：

```text
holdout median pixel error + 0.25 × holdout P95 pixel error
```

两者在 `0.05 px` 内近似相同时优先 pinhole，避免为无法验证的微小训练误差引入更复杂模型。输出包括：

- `camera.npz`：安全的 NumPy 标定文件；
- `camera.report.json`：训练 RMS、留出 median/P95/max、成功 fold 和选择依据。

显式指定 `pinhole` 或 `fisheye` 时仍允许 3–5 张有效图，但报告不会伪造留出误差；生产标定仍应补足视图。

## 3. 标注多个 pan 锚帧

使用 [球场注册真值标注流程](PITCH_REGISTRATION_ANNOTATION.md) 对至少两个不同水平角度的清晰帧标注球场对应点。推荐左、中、右各 1–3 帧，每帧至少 8 个点并覆盖两个非平行方向族。

如果硬件能提供编码器角度，可在帧对象中加入：

```json
{"encoder_pan_rad": 0.123}
```

没有硬件角度也可以从视频对应点估计相对 pan。

## 4. 生成 CameraRigProfile

```bash
uv run python tools/calibrate_camera_rig.py \
  --annotations /path/to/left.json /path/to/centre.json /path/to/right.json \
  --lens assets/calibration/camera.npz \
  --camera-id stadium-main-1 \
  --output assets/calibration/stadium-main-1.rig.json \
  --pan-min-deg -70 \
  --pan-max-deg 70
```

工具先把人工点统一变换到去畸变后的 pinhole 像素系，再估计固定相机中心、基础姿态和各锚帧相对 pan。默认质量门禁：

- 锚帧估计的相机中心最大离散不超过 `0.5 m`；
- 任一锚帧平均重投影误差不超过 `3 px`。

失败时只保存报告、不保存可用于运行的 profile。不要通过放宽阈值掩盖错点、错误球场尺寸或不正确的畸变模型。

## 5. 真实验收

合成单元测试只能验证几何实现。生产前必须额外检查：

- 留出棋盘格 median/P95；
- 去畸变后画面边缘的真实直线残差；
- 多 pan 锚帧相机中心离散；
- 独立球场点的重投影误差；
- 左右边缘是否仍存在方向一致的径向残差；
- 快速 pan 视频中的模板跳变与恢复时间。
