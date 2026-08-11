# 任务 A：完成 test1 球场与球员接触点标注

## 0. 你领取的任务是什么

你负责把一段足球视频 `test1.mp4` 变成一份可用于开发和误差分析的人工真值集。

你不需要训练模型，不需要运行 YOLO，不需要 NVIDIA GPU，也不需要理解整个项目。你只需要：

```text
安装 Git 和 uv
→ 从 GitHub 获取标注工具
→ 从 test1.mp4 自动抽取 30 帧
→ 在浏览器点击球场白线交点和球员脚底
→ 运行自动校验
→ 提交 JSON、报告和复核记录
```

预计用时：环境准备 15–30 分钟，标注 60–120 分钟，复核 20–40 分钟。

## 1. 开始前向负责人索取

没有以下内容不要开始：

- [ ] `test1.mp4` 原始文件；
- [ ] 本次球场实际长宽；不知道时由负责人明确允许使用 `105×68 m`；
- [ ] 你的 GitHub 账号已能访问 `REDACTED/SC`；
- [ ] 一名负责抽查至少 5 帧的复核人员。

视频通过网盘、移动硬盘或局域网传输，不要从聊天软件二次压缩后再保存。

## 2. 安装基础工具

### 2.1 安装 Git

先在终端执行：

```bash
git --version
```

能看到版本号即可。如果提示找不到命令，安装 [Git](https://git-scm.com/downloads) 后重新打开终端。

### 2.2 安装 uv

macOS/Linux：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

重新打开终端，确认：

```bash
uv --version
```

本任务使用 `uv run --no-project`，只下载 NumPy 和 OpenCV，不会安装 PyTorch、CUDA 或完整模型环境。

## 3. 从 GitHub 建立自己的工作分支

以下命令中的 `<你的名字>` 改成简短英文或 GitHub 用户名，例如 `alice`。

```bash
git clone https://github.com/refereelink-team/refereelink-backend.git
cd SC
git fetch origin
git switch --create data/test1-pitch-<你的名字> origin/codex/dev-pitch-projection
```

检查当前分支：

```bash
git branch --show-current
```

输出应类似：

```text
data/test1-pitch-alice
```

如果 `origin/codex/dev-pitch-projection` 不存在，停止操作并联系负责人，不要改用 `main`。

## 4. 放置视频并记录来源哈希

在仓库根目录执行：

```bash
mkdir -p task_data
```

Windows PowerShell 可使用：

```powershell
New-Item -ItemType Directory -Force task_data
```

把视频复制成：

```text
SC/task_data/test1.mp4
```

计算 SHA-256。

macOS/Linux：

```bash
shasum -a 256 task_data/test1.mp4
```

Windows PowerShell：

```powershell
Get-FileHash task_data/test1.mp4 -Algorithm SHA256
```

把结果保存下来。若负责人提供了标准哈希，必须完全一致；不一致时不要继续标注。

## 5. 自动生成 30 帧标注包

下面假设球场尺寸为 105×68 m。如果负责人给了其他尺寸，替换两个数字。

```bash
uv run --no-project \
  --with "numpy>=1.26" \
  --with "opencv-python>=4.8" \
  python -m tools.create_pitch_annotation_pack \
  task_data/test1.mp4 \
  task_data/test1-pack \
  --frames 30 \
  --split calibration \
  --pitch-length-m 105 \
  --pitch-width-m 68
```

PowerShell 可以把反斜杠换行删除，整条命令写成一行。

成功标志：

```text
Created annotation pack: .../task_data/test1-pack/manifest.json
```

并且目录中存在：

```text
task_data/test1-pack/
├── frames/          30 张 JPG
├── index.html
├── manifest.json
└── README.txt
```

先检查空包结构：

```bash
uv run --no-project \
  --with "numpy>=1.26" \
  --with "opencv-python>=4.8" \
  python -m tools.validate_pitch_annotations \
  task_data/test1-pack/manifest.json \
  --allow-empty
```

退出时没有报错即表示工具和帧文件正常。

## 6. 打开标注页面

在仓库根目录开启本地网页服务：

```bash
uv run --no-project python -m http.server 8765 \
  --bind 127.0.0.1 \
  --directory task_data/test1-pack
```

这个终端要保持打开。浏览器访问：

```text
http://127.0.0.1:8765/index.html
```

正常界面参考：

![标注界面示例](../assets/pitch-registration-annotation-implementation.png)

如果页面空白：

1. 确认终端中的 HTTP 服务仍在运行；
2. 确认 URL 是 `127.0.0.1:8765/index.html`；
3. 确认 `task_data/test1-pack/frames/` 中有 JPG；
4. 按 `Ctrl+C` 关闭服务后重新运行。

## 7. 先理解两类点

### 7.1 蓝色点：球场锚点

球场锚点是确定白线位置的点。模式选择 `Pitch landmarks` 或按键盘 `1`。

统一坐标方向：

```text
远端边线：top，Y=0

左侧球门线                             右侧球门线
left，X=0  ├──────── X 增大 ────────┤ right，X=105

相机侧边线：bottom，Y=68
                     [主机位相机]
```

`left/right` 永远指整座球场的左、右球门，不能因为某一帧画面转向就交换。

常见名称：

| 画面内容 | 优先选择的标签 |
|---|---|
| 中线与远端边线交点 | `centre_top` |
| 中线与近端边线交点 | `centre_bottom` |
| 中线与中圈上/下交点 | `centre_circle_top` / `centre_circle_bottom` |
| 中圈最左/最右点 | `centre_circle_left` / `centre_circle_right` |
| 中点 | `centre_spot` |
| 左侧禁区前沿上/下角 | `left_penalty_area_front_top` / `left_penalty_area_front_bottom` |
| 右侧禁区前沿上/下角 | `right_penalty_area_front_top` / `right_penalty_area_front_bottom` |
| 左/右球场四角 | `left_top_corner` 等对应名称 |

每帧操作：

1. 在右侧搜索并选择一个标签；
2. 点击对应白线中心的交点；
3. 再选下一个标签；
4. 至少标 4 个，推荐 8–12 个；
5. 点应覆盖画面多个方向，不能都落在同一条线；
6. 看不清、被遮挡或只能猜的位置不要标。

### 7.2 黄色点：球员地面接触点

切换到 `Contact points` 或按 `2`。先输入 ID，再点击脚底。

点击定义：

- 双脚都着地：点击两只鞋与草地接触位置的中点；
- 一只脚着地：点击着地脚鞋底；
- 跳起、倒地、脚被挡住或画面截断：跳过；
- 可以标普通球员、守门员和裁判；
- 不标足球、观众、替补席和场外工作人员。

ID 不确定时使用：

```text
f<源帧号>-p<本帧序号>
```

例如 `f0123-p01`。不要求跨帧猜同一个人。

本任务目标：全包推荐 80–100 个清晰接触点，每帧通常 3–8 个。

## 8. 每帧的固定操作顺序

对 30 帧逐帧执行：

1. 按 `1`，标 8–12 个可靠球场锚点；
2. 确认至少包含两个不平行方向；
3. 按 `2`，标所有清晰可见的脚底接触点；
4. 检查是否误点到腿、鞋面、阴影或白线边缘；
5. 每 5 帧点击 `Save draft`；
6. 按 `→` 进入下一帧。

快捷键：

| 操作 | 快捷键 |
|---|---|
| 球场锚点 | `1` |
| 接触点 | `2` |
| 上/下一帧 | `←` / `→` |
| 撤销 | `Ctrl+Z` 或 `Command+Z` |

注意：`Save draft` 只保存在当前浏览器。任务完成必须点击 `Export JSON`。

## 9. 导出并运行校验

点击右上角 `Export JSON`。浏览器会下载类似：

```text
pitch-annotations-test1.mp4.json
```

把它复制回标注包并重命名。

macOS/Linux：

```bash
cp ~/Downloads/pitch-annotations-test1.mp4.json \
  task_data/test1-pack/manifest.annotated.json
```

Windows PowerShell：

```powershell
Copy-Item "$HOME\Downloads\pitch-annotations-test1.mp4.json" `
  "task_data\test1-pack\manifest.annotated.json"
```

执行正式校验：

```bash
uv run --no-project \
  --with "numpy>=1.26" \
  --with "opencv-python>=4.8" \
  python -m tools.validate_pitch_annotations \
  task_data/test1-pack/manifest.annotated.json \
  --output task_data/test1-pack/validation-report.json
```

打开 `validation-report.json`。完成标准：

```json
{
  "valid": true,
  "issues": []
}
```

还应确认：

- `split` 是 `calibration`；
- `frame_count` 是 30；
- `annotated_frame_count` 大于 0；
- `contact_count` 推荐达到 80–100；
- 每个 `frames[i].issues` 都是空数组。

常见错误：

| 错误 | 怎么修 |
|---|---|
| `fewer_than_four_correspondences` | 该帧补到至少 4 个锚点，或删除该帧全部不可靠锚点 |
| `degenerate_correspondences` | 点接近共线，补另一方向的交点 |
| `insufficient_image_coverage` | 点都挤在局部，增加画面其他区域的可靠点 |
| `high_landmark_residual` | 多半是标签选错、左右反了或点击偏离交点，逐点复查 |
| `missing_frame_image` | JSON 没复制回 `test1-pack` 目录 |

修正后重新 `Export JSON`，覆盖 `manifest.annotated.json`，再次执行校验。

## 10. 第二人复核

复核人至少检查 5 帧，优先选择：

- 锚点最多的帧；
- 画面边缘畸变明显的帧；
- 球员遮挡多的帧；
- 左右球门方向容易混淆的帧；
- 随机抽取一帧。

建议标准：同一球场点两人差异尽量不超过 2 px，脚底点尽量不超过 4 px。复核发现问题后，由主标注员修改并重新校验。

## 11. 准备 GitHub 交付物

原始视频和 `frames/` 不提交 Git。创建交付目录：

```bash
mkdir -p submissions/pitch-registration/test1
cp task_data/test1-pack/manifest.annotated.json \
  submissions/pitch-registration/test1/
cp task_data/test1-pack/validation-report.json \
  submissions/pitch-registration/test1/
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force submissions\pitch-registration\test1
Copy-Item task_data\test1-pack\manifest.annotated.json `
  submissions\pitch-registration\test1\
Copy-Item task_data\test1-pack\validation-report.json `
  submissions\pitch-registration\test1\
```

创建 `submissions/pitch-registration/test1/review-notes.md`，内容至少包括：

```markdown
# test1 标注交付

- 标注人：
- 复核人：
- 视频 SHA-256：
- 球场尺寸：
- 标注日期：
- 球场锚点难点：
- 接触点跳过规则是否一致：是/否
- 复核帧号：
- validation valid：true
- contact_count：
- 原始完整标注包存放位置：网盘链接或交接方式
```

提交：

```bash
git add submissions/pitch-registration/test1
git commit -m "data(field): annotate test1 calibration set"
git push -u origin HEAD
```

在 GitHub 创建 PR：

- base：`codex/dev-pitch-projection`；
- compare：你的 `data/test1-pitch-<名字>`；
- 标题：`data(field): annotate test1 calibration set`；
- PR 中附上完整标注包的交接位置，但不要把原始 MP4 上传进 Git。

## 12. 任务完成清单

- [ ] 使用的是负责人提供的原始 `test1.mp4`，哈希已记录；
- [ ] 从 `codex/dev-pitch-projection` 创建个人分支；
- [ ] 生成了 30 帧，split=`calibration`；
- [ ] 每个有锚点的帧至少 4 点，推荐 8–12 点；
- [ ] 推荐完成 80–100 个清晰接触点；
- [ ] 没有猜测被遮挡的点；
- [ ] `validation-report.json` 中 `valid=true`、`issues=[]`；
- [ ] 第二人复核至少 5 帧；
- [ ] PR 只提交 JSON、报告和复核记录；
- [ ] PR base 是 `codex/dev-pitch-projection`。

全部勾选后，任务 A 才算完成。
