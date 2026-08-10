# 球场注册真值标注流程

该工具用于建立球场注册与球员地面接触点的独立真值集。标注结果用于评估，不能与训练/标定序列混用。

## 1. 生成标注包

```bash
uv run python tools/create_pitch_annotation_pack.py \
  /path/to/match.mp4 \
  /path/to/output-pack \
  --frames 30 \
  --start-sec 0 \
  --end-sec 60 \
  --split test \
  --pitch-length-m 105 \
  --pitch-width-m 68
```

一个完整视频序列只能属于 `calibration`、`train`、`validation` 或 `test` 中的一个集合。禁止从同一连续片段抽取相邻帧后分别放入训练集和测试集。

## 2. 打开标注页

```bash
cd /path/to/output-pack
uv run python -m http.server 8765 --bind 127.0.0.1
```

打开 `http://127.0.0.1:8765/index.html`。

- `Pitch landmarks`：先在右侧选择球场点位，再点击画面中的对应像素；
- `Contact points`：输入稳定的球员或 Track ID，再点击双脚之间的地面支撑中心；
- 每帧至少标 4 个不接近共线的球场点，推荐 8 个以上并覆盖尽可能大的画面区域；
- 看不清、被完全遮挡或脚部被画面裁切时，不猜测接触点；
- `Save draft` 保存到当前浏览器本地草稿；
- `Export JSON` 下载可提交给验证器的标注文件。

快捷键：`1` 切换球场点，`2` 切换接触点，左右方向键翻帧，`Cmd/Ctrl+Z` 撤销。

## 3. 验证

```bash
uv run python tools/validate_pitch_annotations.py \
  /path/to/exported-annotations.json \
  --output /path/to/validation-report.json
```

验证器检查：

- JSON 版本和数据类型；
- 图片是否存在；
- 点位标签是否属于统一的 32 点球场模型；
- 坐标是否有限且位于合法数据结构中；
- 点数、图像覆盖度和几何退化；
- MAGSAC 拟合后的人工点残差；
- 接触点数量。

新生成但尚未标注的空包可以使用 `--allow-empty` 检查文件完整性。

## 4. 评估数据合并

推理程序应把每帧预测写入同一 JSON：

```json
{
  "predicted_image_to_pitch": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
  "status": "corrected",
  "latency_ms": 2.3,
  "contact_points": [
    {
      "contact_id": "track-20",
      "image_xy": [706.0, 451.0],
      "predicted_image_xy": [704.0, 453.0]
    }
  ]
}
```

然后执行：

```bash
uv run python tools/evaluate_pitch_registration.py \
  evaluation-input.json \
  --output evaluation-report.json
```

仅有稀疏人工对应点时，评估器仍会报告像素重投影误差和米制投影误差；只有提供独立的稠密真值矩阵时才报告全图网格误差。`homography_available_ratio` 不能代替准确率。

## 5. 标注数量

阶段 A 最低目标：

- `test1.mp4`、`test2.mp4` 和真实固定云台视频各选择困难片段；
- 每类片段覆盖静止、慢速 pan、快速 pan、遮挡和画面边缘；
- 球场锚帧按完整视频序列划分数据集；
- 至少 200 个球员地面接触点；
- 训练/验证/测试中的比赛或连续序列互不重叠。
