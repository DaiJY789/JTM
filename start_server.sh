#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$PROJECT_DIR/logs/server.pid"
CONSOLE_LOG="$PROJECT_DIR/logs/server_console.log"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_DIR/.venv/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON="$CONDA_PREFIX/bin/python"
else
  echo "启动失败：没有找到可用的 Python 环境。"
  echo "请先激活 Conda 环境，或通过 PYTHON_BIN=/path/to/python 指定解释器。"
  exit 1
fi

if ! "$PYTHON" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "启动失败：当前 Python 环境缺少 FastAPI/Uvicorn 依赖：$PYTHON"
  echo "请执行：python -m pip install -r requirements-deploy.txt"
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(<"$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "服务已经在运行，PID: $old_pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

cd "$PROJECT_DIR"
nohup "$PYTHON" -m app.main >>"$CONSOLE_LOG" 2>&1 </dev/null &
server_pid=$!
echo "$server_pid" >"$PID_FILE"

sleep 1
if kill -0 "$server_pid" 2>/dev/null; then
  echo "服务启动成功，PID: $server_pid"
  echo "Python: $PYTHON"
  echo "地址和端口由 .env 中的 APP_HOST、APP_PORT 配置。"
  echo "控制台日志: $CONSOLE_LOG"
else
  rm -f "$PID_FILE"
  echo "启动失败，请查看日志: $CONSOLE_LOG"
  exit 1
fi
