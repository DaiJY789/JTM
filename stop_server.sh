#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$PROJECT_DIR/logs/server.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "服务未运行：没有找到 PID 文件。"
  exit 0
fi

server_pid="$(<"$PID_FILE")"
if [[ ! "$server_pid" =~ ^[0-9]+$ ]]; then
  echo "停止失败：PID 文件内容无效。"
  exit 1
fi

if ! kill -0 "$server_pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "服务已经停止，已清理过期 PID 文件。"
  exit 0
fi

kill "$server_pid"
for _ in {1..10}; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "服务已停止。"
    exit 0
  fi
  sleep 1
done

echo "服务未能在 10 秒内退出，正在强制停止。"
kill -KILL "$server_pid" 2>/dev/null || true
rm -f "$PID_FILE"
echo "服务已停止。"
