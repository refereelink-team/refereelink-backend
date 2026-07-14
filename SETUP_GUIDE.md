# 快速启动指南

## 1. 环境状态

Python 虚拟环境: `.venv/` (已就绪)
Node.js: v20.20.2 (通过 nvm 管理)
CUDA: 可用 (torch 2.13.0+cu130)

## 2. 模型权重 (需手动下载)

当前网络无法访问 Google Drive，请从以下链接手动下载 3 个 `.pt` 文件，放入 `assets/weights/`:

```bash
mkdir -p assets/weights
```

| 模型 | 下载链接 | 放入路径 |
|------|----------|----------|
| 足球检测 | https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V | `assets/weights/football-ball-detection.pt` |
| 球员检测 | https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q | `assets/weights/football-player-detection.pt` |
| 球场检测 | https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf | `assets/weights/football-pitch-detection.pt` |

可选：犯规检测模型 `mvfoul.pth.tar` (需要 fouls_far 库支持)

## 3. 示例视频 (需手动下载)

| 视频 | 下载链接 | 放入路径 |
|------|----------|----------|
| sample 1 | https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF | `assets/data/0bfacc_0.mp4` |
| sample 2 | https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf | `assets/data/2e57b9_0.mp4` |
| sample 3 | https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK- | `assets/data/08fd33_0.mp4` |
| sample 4 | https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU | `assets/data/573e61_0.mp4` |
| sample 5 | https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu | `assets/data/121364_0.mp4` |

> 也可以使用任意本地 `.mp4` 文件，不需要必须下载这些示例视频。

## 4. 启动后端 (终端 1)

```bash
cd /home/caysonyin/Projects/SC
source .venv/bin/activate

# 方式 A: 空载启动 (推荐，从前端配置视频源)
python -m app.server.main --device cuda --port 8000

# 方式 B: 直接指定视频源启动
python -m app.server.main --video_source assets/data/2e57b9_0.mp4 --device cuda --port 8000
```

后端启动后会监听:
- `http://localhost:8000` — FastAPI 主服务
- `http://localhost:8000/video/stream` — MJPEG 视频流
- `ws://localhost:8000/ws/state` — WebSocket 状态推送

## 5. 启动前端 (终端 2)

```bash
cd /home/caysonyin/Projects/SC/web
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
npm run dev
```

打开浏览器: `http://localhost:5173`

## 6. 使用步骤

1. 浏览器打开 `http://localhost:5173`
2. 在 CONTROL 面板的 **VIDEO SOURCE** 输入框填入视频路径，例如:
   - 本地文件: `assets/data/2e57b9_0.mp4`
   - RTSP 摄像头: `rtsp://192.168.1.100:554/stream`
3. 选择 **DEVICE** (`cpu` 或 `cuda`)
4. 勾选需要的选项 (Foul Detect, Show Keypoints)
5. 点击 **START**
6. 视频开始播放，右侧显示 2D 球场投影、状态卡片、事件告警

## 7. 常见问题

**Q: 点击 START 后显示 "video source not found"**
A: 路径不对。用绝对路径，例如 `/home/caysonyin/Projects/SC/assets/data/2e57b9_0.mp4`

**Q: 点击 START 后后端日志显示 YOLO 下载进度条**
A: 模型权重 `.pt` 文件缺失。请按步骤 2 下载并放入 `assets/weights/`。

**Q: RTSP 摄像头没有画面**
A: 检查 RTSP URL 是否可达，摄像头是否在线。后端会自动重连 10 次。

**Q: 前端显示 "WS OFF"**
A: 后端没有启动，或者端口不对。检查 `http://localhost:8000/health` 是否返回 `{"status": "ok"}`。
