# 多视角“位置 + 事实 + 规则”辅助判罚架构

## 系统边界

本模块是人工复核工具，不是全自动裁判。四层职责严格分开：

```text
MViT / Grad-CAM     视觉建议与模型关注证据
人工审核             犯规位置与关键事件事实
球场几何             半场、禁区和犯规方本方禁区关系
IFAB 规则引擎        重启方式、纪律处罚和规则路径
```

MViT 的动作和牌级输出不会直接成为最终判罚。Grad-CAM 只解释模型在什么时间、
什么画面区域产生了较强响应，不表示接触真值或球场坐标。

## 审核流程

1. 运行 `/api/multiview/analyze`，得到模型 Top-K、视角注意力和 Grad-CAM。
2. 在标准 `105 × 68 m` 球场上点击或拖动设置犯规点。
3. 渐进式确认发生犯规、动作、球队、比赛状态、接触、强度和战术影响。
4. 保存审核。服务端重新计算球场几何和 IFAB 规则，前端不能提交判罚结论。
5. 可选调用解释接口。本地 LLM 只能组织措辞；校验失败时退回确定性模板。
6. 每次保存生成新修订。刷新页面或重启后可恢复，历史记录不可原地覆盖。

## API

```text
POST /api/multiview/analyze
GET  /api/multiview/cases/{case_id}/review
PUT  /api/multiview/cases/{case_id}/review
GET  /api/multiview/cases/{case_id}/review/history
POST /api/multiview/cases/{case_id}/explanation
```

`PUT review` 必须携带 `expected_revision`。版本落后时返回 HTTP 409 和当前修订号，
防止多人复核时静默覆盖。

## 数据与配置

默认 SQLite 文件：

```text
var/multiview/reviews.sqlite3
```

可通过以下变量覆盖：

```text
SC_MULTIVIEW_REVIEW_DB
SC_EXPLANATION_LLM_URL
SC_EXPLANATION_LLM_MODEL
SC_EXPLANATION_LLM_TIMEOUT_S
```

数据库保存分析快照和追加式审核修订，不修改 `assets/multiview/cases.json`。数据库、
视频、外部模型源码和权重均不提交 Git。

## 首版规则范围

规则集版本为 `IFAB_2026_27`，覆盖当前 MVFoul 动作相关的接触犯规、模拟、
草率/鲁莽/使用过分力量、SPA、DOGSO、直接任意球、间接任意球和点球。

规则引擎遵循以下安全约束：

- 未确认事实不会被模型默认值补齐；
- 缺少犯规方或防守方向时，不声称禁区点球条件成立；
- 部分事实足以确定牌级时可以输出部分结论，但状态为 `incomplete`；
- 超出首版范围的动作返回 `unsupported`；
- 球已不在比赛中时维持原恢复方式，但仍可给出纪律处罚；
- 规则结果包含使用事实、Law/section、优先级和缺失事实。

首版不自动处理手球、越位、优势原则、多次连续犯规、场外事件或完整 DOGSO
四要素视觉识别。这些场景必须由人工裁判处理。

## 本地解释模型

解释服务兼容独立 `llama-server` 的 OpenAI 风格接口。推荐运行 Qwen3-4B 的 Q4
量化版本，但主应用不会自动下载或启动模型。输入仅包含已确认事实、模型证据快照和
规则结果；响应必须原样返回规则引擎确定的 `restart`、`sanction` 和已知 `rule_ids`。

若服务未配置、超时、JSON 不合法、引用未知规则或试图改变牌级/重启方式，接口返回
确定性模板，并在 `fallback_reason` 中记录原因。

## 验证命令

```bash
uv run pytest -q
uv run ruff check app/multiview app/server/api/multiview.py tests/test_multiview_*.py
cd web && npm run build
```

远程 CUDA 验收使用 SoccerNet `action_46`、`action_144` 和 `action_90`，分别覆盖
黄牌、红牌和不犯规路径。规则引擎不增加 MViT 前向次数。
