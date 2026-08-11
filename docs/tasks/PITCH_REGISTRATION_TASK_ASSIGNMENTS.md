# 球场注册人工任务分配入口

这不是算法设计文档，而是三位组员领取任务时的入口。每个人只需要阅读自己的一份任务书。

## 负责人分配前先做四件事

1. 把任务书链接发给对应组员；
2. 把 `test1.mp4` 或 `test2.mp4` 单独发给标注人员，视频不在 Git 仓库中；
3. 告诉组员本次使用的球场尺寸。未知时明确回复“先按 105×68 m”；
4. 在 GitHub 上确认 `codex/dev-pitch-projection` 分支可见，并允许组员创建分支和 PR。

## 三个独立任务

| 任务 | 适合人员 | 是否需要 NVIDIA GPU | 任务书 |
|---|---|---:|---|
| A：test1 开发校准集 | 无硬件同伴 1 | 否 | [TASK_A_TEST1_ANNOTATION.md](TASK_A_TEST1_ANNOTATION.md) |
| B：test2 锁定测试集 | 无硬件同伴 2 | 否 | [TASK_B_TEST2_ANNOTATION.md](TASK_B_TEST2_ANNOTATION.md) |
| C：真实相机与云台 | 有相机、云台和场地条件的同伴 | 否；采集和标定只需 CPU | [TASK_C_CAMERA_AND_PAN_CALIBRATION.md](TASK_C_CAMERA_AND_PAN_CALIBRATION.md) |

三项任务可以并行，不需要等待其他人完成。

## 统一交付规则

- 代码来源：`https://github.com/refereelink-team/refereelink-backend.git`；
- 起始分支：`codex/dev-pitch-projection`；
- 每个人创建自己的工作分支，不直接向开发分支提交；
- PR 的 base 选择 `codex/dev-pitch-projection`，不是 `main`；
- 原始 MP4、棋盘格照片和云台视频不提交 Git；
- JSON、校验报告、配置、哈希和复核记录可以提交；
- 看不清的点一律跳过，不允许猜测；
- 命令失败时保留完整终端输出，不要只截最后一行。

## 总体完成顺序

```text
A 完成 test1 ─┐
              ├─→ 主开发者冻结当前 P0 误差基线
B 完成 test2 ─┘

C 完成 camera.npz
      ↓
C 完成多 pan 锚帧
      ↓
C 生成 camera-rig.json
      ↓
主开发者进行真实物理 pan CUDA 验收
```

## 主开发者验收时只看这些

### 任务 A

- `split` 必须是 `calibration`；
- 30 帧中每个用于几何评估的帧至少 4 个锚点，推荐 8–12 个；
- 推荐 80–100 个清晰接触点；
- `validation-report.json` 中 `valid=true`、`issues=[]`。

### 任务 B

- `split` 必须是 `test`；
- 标注期间没有用 test2 运行模型、调参或选择方案；
- 推荐 80–100 个清晰接触点；
- 文件已计算 SHA-256，提交后冻结。

### 任务 C

- 棋盘格有效图片推荐不少于 20 张；
- `camera.report.json` 已完成 pinhole/fisheye 留出比较；
- 多 pan 锚帧来自最终安装位置、固定焦距和固定俯仰；
- `camera-rig.report.json` 中 `accepted=true`、`issues=[]`。
