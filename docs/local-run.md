# 本地运行

[返回项目首页](../README.md)

以下说明适用于希望在自己的环境中运行惊堂木的用户。在线体验请使用项目首页提供的 Demo 地址。


使用 Python 3.10 或 3.11。以下命令适用于 macOS / Linux：

```bash
git clone https://github.com/DaiJY789/JTM.git
cd JTM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，选择模型服务并填写对应密钥。使用 DashScope 时，至少配置：

```dotenv
MODEL_PROVIDER=dashscope
DASHSCOPE_API_KEY=你的模型服务密钥
APP_HOST=127.0.0.1
APP_PORT=5000
```

仅配置模型接口也可发起检测。需要外部搜索证据时，填写 `BOCHA_API_KEY`；未配置搜索服务时，结果不含该服务提供的外部候选材料，应特别关注证据局限。其他模型服务、模型名称及可选设置见 [.env.example](../.env.example)。

优先在一轮搜索与核查后给出判定，必要时补充搜索，总计最多调用工具 3 次（失败调用也计入）。可通过 `MFC_MAX_TOOL_ROUNDS` 调整工具调用轮数（最多 3 轮）、`MFC_SEARCH_QUERY_COUNT` 调整总搜索次数（最多 3 次），通过 `MFC_AGENT_TIMEOUT_SECONDS` 设置整体时间预算（默认 120 秒）。达到预算后将要求模型结束核查；整体超时或模型无法返回有效最终结果时显示检测失败。

相关配置统一使用 `MFC_*` 命名；已有 `MMD_*` 环境变量仍可读取，同时配置时优先使用 `MFC_*`。

启动工作台：

```bash
python -m app.main
```

- 工作台：<http://127.0.0.1:5000/>
- 接口文档：<http://127.0.0.1:5000/docs>

也可使用脚本在后台运行：

```bash
bash start_server.sh
bash stop_server.sh
```

启动脚本优先使用项目的 `.venv`，也支持当前已激活的 Conda 环境或通过 `PYTHON_BIN` 指定解释器。

更新版本时，先停止服务。前台运行可按 `Ctrl+C`；后台运行可使用停止脚本。保留自己的 `.env`，然后更新代码和依赖：

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.main
```

若原先在后台运行，最后一步改为 `bash start_server.sh`。修改代码或 `.env` 后也需要重启服务，并刷新页面加载新版界面；已有历史结论不会自动重新检测。
