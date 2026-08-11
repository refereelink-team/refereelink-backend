# 任务 B：完成 test2 锁定测试集标注

## 0. 你领取的任务是什么

你负责为 `test2.mp4` 建立一份独立的、最终用于报告准确率的测试真值。

这份数据和普通开发数据最大的区别是：

> 你只负责按照统一规则标注，不能根据任何模型结果修改标法，也不能让其他人用它调参数、挑算法或训练模型。

你不需要 NVIDIA GPU，不需要安装 CUDA，不需要运行 YOLO。完整流程是：

```text
安装 Git 和 uv
→ 从 GitHub 获取工具
→ 从 test2.mp4 抽取 30 帧
→ 独立完成球场锚点和脚底接触点
→ 自动校验
→ 第二人复核
→ 计算哈希并冻结
→ 通过 PR 交付
```

预计用时：环境准备 15–30 分钟，标注 60–120 分钟，复核和冻结 30–45 分钟。

## 1. 开始前向负责人索取

- [ ] 未经聊天软件二次压缩的 `test2.mp4`；
- [ ] 视频的标准 SHA-256，或负责人确认由你首次计算并登记；
- [ ] 球场实际长宽；未知时由负责人明确允许按 `105×68 m`；
- [ ] GitHub 仓库访问权限；
- [ ] 一名复核人员；
- [ ] 负责人确认 test2 尚未被作为训练/调参数据使用。

若你已经看过某个算法在 test2 上的投影误差、失败帧排序或模型对比结果，在 `review-notes.md` 中如实说明。不要隐瞒潜在测试泄漏。

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

重新打开终端并检查：

```bash
uv --version
```

任务命令使用轻量环境，只会安装 NumPy 和 OpenCV，不会安装完整 AI 模型环境。

## 3. 克隆代码并创建独立分支

将 `<你的名字>` 替换成英文名或 GitHub 用户名：

```bash
git clone https://github.com/refereelink-team/refereelink-backend.git
cd SC
git fetch origin
git switch --create data/test2-pitch-<你的名字> origin/codex/dev-pitch-projection
git branch --show-current
```

最后一条应输出类似：

```text
data/test2-pitch-bob
```

如果找不到 `origin/codex/dev-pitch-projection`，停止并联系负责人。不要从 `main` 或任务 A 的分支开始。

## 4. 放置视频并核对哈希

创建本地数据目录：

macOS/Linux：

```bash
mkdir -p task_data
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force task_data
```

把视频放到：

```text
SC/task_data/test2.mp4
```

计算 SHA-256。

macOS/Linux：

```bash
shasum -a 256 task_data/test2.mp4
```

Windows PowerShell：

```powershell
Get-FileHash task_data/test2.mp4 -Algorithm SHA256
```

与负责人登记值不一致时立刻停止。测试视频版本不同会让最终准确率失去可比性。

## 5. 创建锁定测试标注包

以下命令中的 `--split test` 不能改成 calibration 或 validation。

```bash
uv run --no-project \
  --with "numpy>=1.26" \
  --with "opencv-python>=4.8" \
  python -m tools.create_pitch_annotation_pack \
  task_data/test2.mp4 \
  task_data/test2-pack \
  --frames 30 \
  --split test \
  --pitch-length-m 105 \
  --pitch-width-m 68
```

PowerShell 可以删除反斜杠换行，将命令写成一行。

成功时会出现：

```text
Created annotation pack: .../task_data/test2-pack/manifest.json
```

目录应包含 30 张 JPG、`index.html` 和 `manifest.json`。

验证空包：

```bash
uv run --no-project \
  --with "numpy>=1.26" \
  --with "opencv-python>=4.8" \
  python -m tools.validate_pitch_annotations \
  task_data/test2-pack/manifest.json \
  --allow-empty
```

无报错即可以开始标注。

## 6. 打开浏览器标注工具

```bash
uv run --no-project python -m http.server 8765 \
  --bind 127.0.0.1 \
  --directory task_data/test2-pack
```

保持终端窗口运行，在浏览器打开：

```text
http://127.0.0.1:8765/index.html
```

正常界面参考：

![标注界面示例](../assets/pitch-registration-annotation-implementation.png)

若画面加载失败，检查 `task_data/test2-pack/frames/` 是否有 30 张图片，并重启 HTTP 服务。

## 7. 球场锚点标注规则

按 `1` 进入 `Pitch landmarks`。

坐标方向固定如下：

```text
远端边线 = top = Y=0

left goal，X=0  ├────── X 向右增加 ──────┤ right goal，X=105

相机侧边线 = bottom = Y=68
                         [相机]
```

这些名称描述的是整座球场，不是某一帧屏幕临时的左右。镜头转向后也不能交换 `left` 和 `right`。

常见标签：

| 白线位置 | 标签 |
|---|---|
| 中线与远端/近端边线交点 | `centre_top` / `centre_bottom` |
| 中线与中圈上/下交点 | `centre_circle_top` / `centre_circle_bottom` |
| 中圈最左/最右点 | `centre_circle_left` / `centre_circle_right` |
| 中点 | `centre_spot` |
| 左禁区前沿上/下角 | `left_penalty_area_front_top` / `left_penalty_area_front_bottom` |
| 右禁区前沿上/下角 | `right_penalty_area_front_top` / `right_penalty_area_front_bottom` |
| 四个球场角 | `left_top_corner`、`left_bottom_corner`、`right_top_corner`、`right_bottom_corner` |

每个用于几何评估的帧：

- 最少 4 点，推荐 8–12 点；
- 至少覆盖两组不平行白线；
- 尽量覆盖画面左、右、上、下；
- 点击白线中心或交点中心；
- 被球员挡住、过度模糊、只能依靠经验猜的位置不要标；
- 同一条直线上的很多点不能替代二维覆盖。

## 8. 球员接触点标注规则

按 `2` 进入 `Contact points`，先输入 ID，再点击脚底支撑中心。

| 场景 | 点击位置 |
|---|---|
| 双脚着地 | 两个鞋底接地点中点 |
| 只有一脚着地 | 着地脚鞋底接地点 |
| 跑动模糊但脚底仍可辨认 | 两脚最低可见点的中点 |
| 跳起、倒地、脚被挡住、画面裁断 | 跳过 |

可标普通球员、守门员和裁判；不要标足球、工作人员和观众。

不确定人物身份时使用每帧唯一 ID：

```text
f<源帧号>-p<序号>
```

例如 `f1053-p03`。不要根据同色球衣强行认为跨帧是同一人。

本任务推荐完成 80–100 个高质量接触点。

## 9. 30 帧逐帧执行流程

每帧按同一顺序：

1. 标 8–12 个可靠球场锚点；
2. 确认不是全部共线或集中在局部；
3. 标所有清晰脚底接触点；
4. 删除任何需要“猜”的点；
5. 每 5 帧点击 `Save draft`；
6. 按右方向键进入下一帧。

快捷键：`1` 球场点、`2` 接触点、`←/→` 切帧、`Ctrl/Cmd+Z` 撤销。

不要打开任务 A 的 manifest 复制点位，也不要运行任何当前球场模型来建议你应该点在哪里。人工规则可以共享，具体坐标必须由你独立判断。

## 10. 导出并校验

完成后点击 `Export JSON`，将下载文件复制回包内：

macOS/Linux：

```bash
cp ~/Downloads/pitch-annotations-test2.mp4.json \
  task_data/test2-pack/manifest.annotated.json
```

Windows PowerShell：

```powershell
Copy-Item "$HOME\Downloads\pitch-annotations-test2.mp4.json" `
  "task_data\test2-pack\manifest.annotated.json"
```

执行：

```bash
uv run --no-project \
  --with "numpy>=1.26" \
  --with "opencv-python>=4.8" \
  python -m tools.validate_pitch_annotations \
  task_data/test2-pack/manifest.annotated.json \
  --output task_data/test2-pack/validation-report.json
```

只有同时满足以下内容才能继续：

```text
valid = true
issues = []
split = test
frame_count = 30
每个 frame 的 issues = []
contact_count 推荐 80–100
```

错误处理：

| 错误 | 修复 |
|---|---|
| `fewer_than_four_correspondences` | 补足可靠锚点，或删除该帧全部不可靠锚点 |
| `degenerate_correspondences` | 增加另一方向的白线交点 |
| `insufficient_image_coverage` | 增加画面其他区域锚点 |
| `high_landmark_residual` | 检查标签方向、点位名称和点击中心 |
| `missing_frame_image` | 把导出 JSON 放回 `test2-pack` 目录 |

修正后必须重新导出并重新校验。

## 11. 复核与冻结

测试集比 calibration 集要求更严格：

1. 第二人至少复核 5 帧；
2. 容易混淆的左/右禁区帧优先复核；
3. 验证通过后计算最终 manifest 的 SHA-256；
4. 把哈希写入复核记录；
5. PR 提交后不再根据模型表现主动修改；
6. 只有发现明确人工错误时才能修正，并在 PR 中记录修改原因和新旧哈希。

macOS/Linux：

```bash
shasum -a 256 task_data/test2-pack/manifest.annotated.json
```

Windows PowerShell：

```powershell
Get-FileHash task_data/test2-pack/manifest.annotated.json -Algorithm SHA256
```

## 12. 准备 GitHub 交付物

```bash
mkdir -p submissions/pitch-registration/test2
cp task_data/test2-pack/manifest.annotated.json \
  submissions/pitch-registration/test2/
cp task_data/test2-pack/validation-report.json \
  submissions/pitch-registration/test2/
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force submissions\pitch-registration\test2
Copy-Item task_data\test2-pack\manifest.annotated.json `
  submissions\pitch-registration\test2\
Copy-Item task_data\test2-pack\validation-report.json `
  submissions\pitch-registration\test2\
```

创建 `submissions/pitch-registration/test2/review-notes.md`：

```markdown
# test2 锁定测试集交付

- 标注人：
- 复核人：
- 原视频 SHA-256：
- manifest SHA-256：
- 球场尺寸：
- 标注日期：
- 复核帧号：
- validation valid：true
- contact_count：
- 标注前是否看过 test2 模型误差或失败排序：是/否；如是请说明
- 是否保证未使用模型输出来调整人工点：是/否
- 完整标注包存放位置：网盘链接或交接方式
```

提交并推送：

```bash
git add submissions/pitch-registration/test2
git commit -m "data(field): add locked test2 ground truth"
git push -u origin HEAD
```

创建 GitHub PR：

- base：`codex/dev-pitch-projection`；
- compare：你的 `data/test2-pitch-<名字>`；
- 标题：`data(field): add locked test2 ground truth`；
- PR 描述明确写“test2 未用于训练、调参或模型选择”；
- 附完整标注包交接位置，不提交原始 MP4。

## 13. 任务完成清单

- [ ] 原视频哈希与负责人记录一致；
- [ ] 从 `codex/dev-pitch-projection` 创建独立分支；
- [ ] 生成 30 帧，split=`test`；
- [ ] 每个几何帧最少 4 个可靠锚点，推荐 8–12 个；
- [ ] 推荐完成 80–100 个清晰接触点；
- [ ] 没有使用模型输出指导或修改人工坐标；
- [ ] `validation-report.json` 中 `valid=true`、`issues=[]`；
- [ ] 第二人复核至少 5 帧；
- [ ] manifest SHA-256 已记录并冻结；
- [ ] PR base 是 `codex/dev-pitch-projection`；
- [ ] 原始视频和 JPG 没有提交 Git。

全部勾选后，任务 B 才算完成。
