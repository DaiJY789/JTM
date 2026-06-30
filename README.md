# 惊堂木：多模态虚假信息检测系统

惊堂木是一个基于 FastAPI、文本大模型和视觉大模型构建的多模态虚假信息检测演示系统，支持中文和英文分析。

系统可以处理以下三类任务：

1. **文本检测**：抽取可核查声明、并行检索证据，完成事实核查或谣言检测。
2. **图像检测**：判断图像属于真实图像、伪造图像还是合成图像，并输出伪造类型和可疑区域。
3. **多模态检测**：并行执行文本检测、图像检测和图文一致性检测，再由模型生成综合结论。

前端采用对话式交互。首次提交后会保留原始文本和图片，后续可以围绕检测结果继续追问。

## 主要功能

### 文本检测

- 最多抽取 3 条可进行事实核查的声明。
- 多条声明并行处理，每条声明并行检索多条证据。
- 根据证据充分程度输出“事实核查”或“谣言检测”类型。
- 生成面向用户的最终判断、置信度和原因分析。

### 图像检测

- 图像三分类：真实图像、伪造图像、合成图像。
- 伪造类型包括：
  - 无
  - 拼接
  - 复制移动
  - 内容移除
  - 内容替换
  - 人脸篡改
  - 局部重绘
  - 全局编辑
  - 未知
- 伪造图像必须输出至少一个可疑区域；合成图像也可以标记局部生成异常。
- 支持多轮工具调用，同一轮中的独立工具可以并行执行。
- 暂停使用 C2PA / Content Credentials 来源凭证检查。
- TruFor、CAT-Net 和 DIRE 等外部模型仅在显式配置命令后启用。

图像分类定义：

- **真实图像**：由真实成像设备直接捕获现实场景，且未经过改变关键语义内容的编辑、篡改或生成操作。压缩、裁剪、缩放、曝光调整等不改变语义事实的处理不影响该分类。
- **伪造图像**：以真实图像为基础，通过插入、删除、替换、拼接、复制、修复、换脸或重绘等操作改变原始视觉语义或事实表达。
- **合成图像**：主体视觉内容主要由生成式 AI、计算机图形渲染、仿真系统或其他合成方式产生，而非由真实相机直接拍摄。

### 多模态检测

多模态模式会并行运行三个分支：

1. 文本真实性检测。
2. 图像真实性检测。
3. 图文一致性检测。

单模态异常综合标签为：

- 文本异常
- 图像异常
- 均无异常
- 全部异常

图文关系类型为：

- 图文一致
- 图文部分一致
- 图文不一致

当图片能够帮助理解文本内容，且不存在明显冲突或误导时，可以判断为图文一致；图片只支持部分内容时判断为图文部分一致。

### 追问与重新检测

- 普通解释性追问默认复用已有结果，不重新运行检索或图像工具。
- 当问题中明确包含“重新检测”“重新分析”“重跑工具”“从头检测”等语义时，系统会重新执行对应检测流程并更新检测结果。
- 追问内容以对话形式展示，最新检测结论会覆盖右侧结果区域。

### 会话历史

- 每个有效对话都会持久化到项目根目录的 `conversations/`。
- 保存内容包括对话历史、最新输入、最新检测结果和上传图片。
- 服务启动时会自动读取已有对话。
- 前端支持恢复和删除历史对话。
- 未输入内容的新对话不会保存。

## 项目结构

```text
app/
├── agents/                 # 文本模型和视觉模型构造
├── api/                    # 检测、追问和会话接口
├── core/                   # 配置、语言与日志
├── prompts/                # 文本、图像和多模态提示词
├── schemas/                # Pydantic 数据结构
├── services/
│   ├── text/               # 声明抽取、证据检索和文本判定
│   ├── vision/             # 图像读取、取证流程和图像判定
│   ├── multimodal/         # 图文一致性与多模态结果融合
│   ├── conversation_history.py
│   ├── detection_orchestrator.py
│   └── followup_analysis.py
├── tools/
│   ├── image/              # 图像取证与辅助工具
│   └── text/               # 文本检索工具
└── web/static/             # 前端页面与静态资源
conversations/              # 对话、结果及上传图片
logs/                       # 运行日志
outputs/vision/             # 图像取证中间结果
tests/                      # 单元测试
config.py                   # 运行配置
requirements.txt            # Python 依赖
```

## 环境要求

- Python 3.10
- 推荐使用 `uv` 管理虚拟环境
- 可访问所配置的大模型、视觉模型和搜索服务
- Redis 为可选依赖，主要流程不应因 Redis 不可用而无法启动

## 安装

```bash
uv venv --python 3.10 .venv
uv pip install -r requirements.txt
uv pip install pytest
```

复制环境变量示例：

```bash
cp .env.example .env
```

至少需要根据实际使用的功能配置以下变量：

```env
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_VISION_MODEL=qwen3-vl-plus

BOCHA_API_KEY=
```

应用通过环境变量读取配置，并会在本地启动时自动加载项目根目录下的 `.env`。也可以继续使用 `--env-file .env` 显式指定环境文件。

不要将真实密钥提交到版本控制系统；生产环境建议使用密钥管理服务。

## 启动项目

```bash
.venv/bin/python -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 5000 \
  --env-file .env
```

启动后访问：

- 前端页面：<http://127.0.0.1:5000>
- OpenAPI 文档：<http://127.0.0.1:5000/docs>

也可以直接运行：

```bash
set -a
source .env
set +a
.venv/bin/python -m app.main
```

此方式会使用 `APP_HOST` 和 `APP_PORT` 配置，默认端口为 `5000`。

## 接口说明

### `POST /detect`

统一检测接口，使用 `multipart/form-data` 提交。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `text` | string | 空 | 新闻文本、事实声明或图文检测中的文本 |
| `image` | file | 空 | 上传的图片 |
| `image_url` | string | 空 | 网络图片地址，与上传图片二选一 |
| `webpage_url` | string | 空 | 新闻网页地址；后端会抽取正文和代表图片进行网页级多模态分析 |
| `question` | string | 自动生成 | 用户问题或检测要求 |
| `mode` | string | `auto` | `auto`、`text_only`、`image_only`、`multimodal` |
| `language` | string | `zh` | `zh` 或 `en` |
| `history` | JSON string | `[]` | 可选的历史消息数组 |
| `conversation_id` | string | 自动生成 | 用于延续已有对话 |

默认问题：

- 仅文本：`请判断这段文字的真实性。`
- 仅图片：`请判断这张图片是真实图像，伪造图像还是合成图像。`
- 图文输入：`请判断这段文字和这张图片是否真实可信，并分析图文是否一致。`

自动模式路由：

- 只有文本时进入 `text_only`。
- 只有图片时进入 `image_only`。
- 同时包含文本和图片时进入 `multimodal`。
- 只有网页时会先抽取网页正文和代表图片；如果网页含正文和图片则进入 `multimodal`，只有正文则进入 `text_only`。
- 显式设置 `mode` 时优先使用指定模式。

文本检测示例：

```bash
curl -X POST http://127.0.0.1:5000/detect \
  -F "mode=text_only" \
  -F "language=zh" \
  -F "text=这里输入需要核查的新闻文本"
```

图像检测示例：

```bash
curl -X POST http://127.0.0.1:5000/detect \
  -F "mode=image_only" \
  -F "language=zh" \
  -F "image=@/path/to/image.jpg"
```

多模态检测示例：

```bash
curl -X POST http://127.0.0.1:5000/detect \
  -F "mode=multimodal" \
  -F "language=zh" \
  -F "text=这里输入新闻文本" \
  -F "image=@/path/to/image.jpg"
```

网页检测示例：

```bash
curl -X POST http://127.0.0.1:5000/detect \
  -F "mode=auto" \
  -F "language=zh" \
  -F "webpage_url=https://example.com/news-page" \
  -F "question=分析这个网页中的文本、图片及图文关系是否真实可信。"
```

### `POST /followup`

基于已有结果继续追问。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `question` | string | 必填，追问内容 |
| `conversation_id` | string | 必填，已有对话 ID |
| `language` | string | `zh` 或 `en` |

```bash
curl -X POST http://127.0.0.1:5000/followup \
  -F "conversation_id=已有的对话ID" \
  -F "language=zh" \
  -F "question=为什么你认为这个区域存在拼接痕迹？"
```

如需重新运行检测，可以明确提问：

```text
请从头重新检测，并重新调用图像工具。
```

### 会话接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/conversations` | 获取历史对话列表 |
| `GET` | `/conversations/{conversation_id}` | 获取指定对话 |
| `DELETE` | `/conversations/{conversation_id}` | 删除指定对话及其图片 |
| `GET` | `/conversations/{conversation_id}/assets/{asset_path}` | 读取对话资源 |

## 响应结构

`/detect` 和 `/followup` 返回统一的结构化 JSON。根据检测模式，部分嵌套字段可能为空。

```json
{
  "mode": "multimodal",
  "label": "均无异常",
  "type": "图文一致",
  "summary": "文本与图像未发现明显异常，图片能够辅助理解文本内容。",
  "question": "请判断这段文字和这张图片是否真实可信，并分析图文是否一致。",
  "confidence": 0.86,
  "runtime_seconds": 12.4,
  "text_verdict": {},
  "image_verdict": {
    "label": "真实图像",
    "forgery_types": ["无"],
    "suspicious_regions": []
  },
  "alignment_verdict": {},
  "evidence": [],
  "claims": [],
  "user_answer": {},
  "conversation_id": "对话ID",
  "conversation_history": [],
  "raw": {}
}
```

常用字段：

- `label`：最终检测标签。
- `type`：文本任务类型、图像伪造类型展示值或图文一致性类型。
- `summary`：综合结论。
- `confidence`：置信度，范围为 0 至 1。
- `runtime_seconds`：本次检测耗时。
- `text_verdict`：文本检测结果和声明证据。
- `image_verdict`：图像分类、伪造类型、工具证据及可疑区域。
- `alignment_verdict`：图文匹配、冲突和缺失信息。
- `user_answer`：供前端展示的最终判断与原因分析。
- `raw`：调试和结构化证据数据。

可疑区域使用原图全局坐标：

```json
{
  "x0": 120,
  "y0": 80,
  "x1": 360,
  "y1": 280,
  "reason": "人物边缘与周围背景衔接不自然"
}
```

## 图像工具

### 伪造取证工具

- `SRM_Filter`：检查局部残差信息是否存在异常。
- `Noise_Analyzer`：检查局部噪声分布是否一致。
- `PatchDense_Scanner`：发现可能的复制移动区域。
- `CFA_Validator`：检查相机成像规律是否存在局部异常。
- `TruFor_Detector`：外部深度模型，可输出整图分数和定位结果；仅在 `VISION_TRUFOR_COMMAND` 已配置时启用。
- `CATNet_Detector`：外部深度模型，用于分析压缩链路异常和篡改定位；仅在 `VISION_CATNET_COMMAND` 已配置时启用。

### 合成检测工具

- `Spectrum_Grid_Analyzer`：检查频率分布和空间异常。
- `Facial_Biological_Auditor`：检查人脸结构、生物特征和物理一致性。
- `DIRE_Detector`：外部模型，用于检测扩散模型生成痕迹；仅在 `VISION_DIRE_COMMAND` 已配置时启用。

### 通用辅助工具

- `C2PA_Provenance_Checker`：当前默认禁用。
- `ROI_Extractor`：裁剪局部区域并保留全局坐标映射。
- `Image_Resizer`：缩放图像并记录比例。
- `Visual_Enhancer`：增强暗区或亮区，辅助观察细节。

## C2PA 配置

当前暂不启用 `C2PA_Provenance_Checker`。以下配置仅作为后续恢复该工具时的参考。

项目可优先使用虚拟环境中的命令：

```text
.venv/bin/c2patool {image_path} --crjson
```

确认工具可用：

```bash
.venv/bin/c2patool --version
```

没有 C2PA 凭证不等于图片伪造，只表示当前文件中未发现可读取的来源声明。

## 外部视觉模型配置

TruFor、CAT-Net 和 DIRE 未配置时不会注册到可调用工具列表，不影响基础视觉分析流程。

```env
VISION_TRUFOR_COMMAND=
VISION_CATNET_COMMAND=
VISION_DIRE_COMMAND=
VISION_EXTERNAL_TOOL_TIMEOUT_SECONDS=120
VISION_MAX_ROUNDS=3
VISION_OUTPUT_DIR=outputs/vision
```

命令模板可使用以下占位符：

- `{image_path}`：输入图片路径。
- `{output_dir}`：本次工具输出目录。
- `{json_path}`：预期的 JSON 结果路径。

外部工具应向标准输出打印 JSON，或将 JSON 写入 `{json_path}`。常用返回字段包括：

```text
score
probability
tamper_score
ai_generated_score
mask_path
heatmap_path
reliability_path
artifacts
candidate_regions
```

## 运行测试

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests
```

只运行某一类测试：

```bash
.venv/bin/python -m pytest tests/test_image_detection_service.py
.venv/bin/python -m pytest tests/test_text_detection_service.py
.venv/bin/python -m pytest tests/test_multimodal_detection_service.py
```

## 运行评测

项目提供了一个最小评测基线，用于把样本、期望标签、运行结果和指标固定下来。

样本文件：

```text
evals/samples/baseline.jsonl
```

默认不调用真实服务，而是使用样本中的 `mock_response` 验证评测脚本本身：

```bash
.venv/bin/python -m evals.run_detection_eval
```

调用本地 `/detect` 服务进行真实评测：

```bash
.venv/bin/python -m evals.run_detection_eval \
  --endpoint http://127.0.0.1:5000/detect \
  --output evals/results/local_baseline.json
```

真实服务模式默认跳过 `live_enabled=false` 的占位样本。当前基线中的图像、网页样本先用于固定评测格式和回归目标，后续替换为本地图片或稳定 URL 后再开启真实评测。

只运行文本样本：

```bash
.venv/bin/python -m evals.run_detection_eval --task text_only
```

列出样本：

```bash
.venv/bin/python -m evals.run_detection_eval --list
```

评测报告会输出：

- `pass_rate`：标签、类型和最低置信度要求全部满足的比例。
- `label_accuracy`：最终标签命中 `accepted_labels` 的比例。
- `family_accuracy`：标签大类命中的比例，用于观察模型方向是否正确。
- `by_task`：按文本、图像、多模态、网页任务拆分的指标。

## 开发说明

- 当前项目用于研究和演示，检测结论不应替代人工核查。
- 搜索结果、模型输出和图像工具都可能存在误差，应结合结构化证据判断。
- 图像工具的异常分数表示可疑程度，不应被单独视为伪造定论。
- 外部模型的运行速度取决于图片大小、模型配置和硬件性能。
- 修改提示词时应同步检查中文与英文版本，并运行对应测试。
