# 多视角犯规判罚页面迁移说明

## 迁移结论

IOT 衍生项目中名为 `MultiViewFoul` 的 React 页面本身仍是占位实现，后端
`multiview/service.py` 也只返回 `not_implemented`。真正可复用的成果是：

1. 静态版“争议事件证据链管理中心”的三栏信息架构与交互；
2. `FoulRecognition` 页的 MViT V2 Small 多视角推理；
3. 对最后一个 MViT transformer block 的 Grad-CAM 定位；
4. Grad-CAM 失败时明确标记为 `optical_flow` 的光流定位；
5. 模型不可用时明确标记为 `scripted` 的演示数据。

本次没有复制 IOT 的占位页面，也没有把外部 GPL 模型代码或大模型权重提交到
SC。实现方式是把上述可用能力按 SC 的 FastAPI + React 架构重新组合。

## 页面入口

启动后访问：

```text
http://localhost:5173/multiview
```

实时分析主页面右上角也提供“多视角判罚”入口。

页面包含：

- 稳定的争议事件队列与状态筛选；
- 主机位、侧机位和反向机位证据；
- 点击辅助机位后提升为主视图；
- 多机位同期时间轴、逐帧与播放演示；
- MViT 判罚建议、置信度、动作类型和牌色；
- Grad-CAM / 光流 / 演示框的来源区分；
- 视角注意力权重；
- 二维球场空间证据；
- 人工确认、暂不确定和归档的会话级操作；
- 模型运行资产缺失原因。

## 后端 API

```text
GET  /api/multiview/cases
GET  /api/multiview/cases/{case_id}
GET  /api/multiview/cases/{case_id}/media/{camera_id}
GET  /api/multiview/status
POST /api/multiview/analyze
```

案例接口不会向浏览器暴露本地文件路径。媒体只能通过已登记的
`case_id + camera_id` 获取。

## 模型链路

真实运行路径：

```text
多视角视频
  -> OpenCV 顺序解码
  -> Kinetics MViT 预处理
  -> SoccerNet VARS MViT_V2_S
  -> 动作分类 + 犯规严重程度
  -> 视角注意力
  -> 最后 transformer block Grad-CAM
  -> 结构化判罚记录
```

Grad-CAM 使用模型最后一个时空 token 网格 `8 x 7 x 7`。去除 CLS token 后，
沿时间维聚合并提取最大高响应连通域，再将 224 crop 坐标映射回原始画面百分比。

Grad-CAM 异常时会尝试 Farneback 光流，但结果的 `localization_source` 会如实写成
`optical_flow`；光流框不能被解释为“模型关注区域”。

## 启用远端 CUDA 真模型

### 1. 同步 Python 环境

```bash
uv sync
```

项目的 `pyproject.toml` 已包含 PyTorch、Torchvision、OpenCV 与 FastAPI。

### 2. 获取官方 VARS 模型代码

```bash
bash tools/setup_mvfoul.sh
```

脚本使用 sparse checkout，仅获取 SoccerNet `sn-mvfoul` 仓库中的
`VARS model` 目录。该目录被 Git 忽略，不进入 SC 源码。

### 3. 放置权重

将官方 `14_model.pth.tar` 放到：

```text
assets/weights/14_model.pth.tar
```

也可以使用环境变量指定已有资产：

```bash
export SC_MVFOUL_CODE_PATH="/path/to/sn-mvfoul/VARS model"
export SC_MVFOUL_WEIGHTS_PATH="/path/to/14_model.pth.tar"
```

### 4. 配置真实多视角案例

默认配置位于 `assets/multiview/cases.json`。真实案例给每个机位增加 `path`：

```json
{
  "camera_id": "main",
  "display_name": "主机位",
  "role": "main",
  "path": "/data/match-001/main.mp4",
  "preview_path": "web/public/multiview/cam_main.jpg",
  "sync_offset_ms": 0
}
```

所有机位都存在有效视频且模型资产就绪时，分析接口自动使用真实模型；否则只有
配置了 `scripted_result` 的案例才能返回演示结果。演示结果始终显示“演示数据”，
不能冒充 CUDA 输出。

若不希望修改仓库配置，可指定外部案例文件：

```bash
export SC_MULTIVIEW_CASES_PATH=/data/sc/multiview-cases.json
```

## 当前边界

- 人工复核状态只保存在当前浏览器会话，尚未写入数据库；
- 默认仓库不包含真实多视角视频、外部模型代码和权重；
- 当前时间轴对静态演示证据做交互模拟，配置真实视频后媒体接口支持浏览器播放；
- MViT 与 Grad-CAM 必须在有官方代码、权重和真实视频的 CUDA 设备上做最终验证；
- 二维球场图目前是证据摘要，后续可接入 SC 动态球场投影输出。

## 验证

```bash
uv run pytest tests/test_multiview_api.py -q
cd web
npm run build
```

自动化测试覆盖案例路径脱敏、媒体服务、显式演示回退、缺失案例和 Grad-CAM
坐标辅助函数。浏览器验收覆盖分析结果、解释框、机位切换、人工复核、主页面入口，
并检查 390 px 窄屏断点无横向溢出。

### RTX 5060 Ti 真实模型验证

本次迁移完成后，在远端 IOT 已有的官方模型资产与 `foul_001` 四机位视频上运行
了新 SC 模块，不修改 IOT/SC 工作树。结果：

```text
GPU: NVIDIA GeForce RTX 5060 Ti
模型: MViT_V2_S
视角数: 4
判定: Tackle / Offence + Yellow Card
置信度: 0.743
视频预处理: 339.0 ms
模型前向: 811.8 ms
Grad-CAM: 239.0 ms
峰值显存: 1267.6 MB
定位来源: gradcam
```

四个视角均生成了合法的百分比定位框，视角注意力为
`[0.0428, 0.3117, 0.3322, 0.3132]`。这证明迁移后的真实 CUDA、分类、视角聚合
和 Grad-CAM 链路可运行；单案例约 1.39 秒属于离线复核路径，不应并入逐帧实时
主管线。
