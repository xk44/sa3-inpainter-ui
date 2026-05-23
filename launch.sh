#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

export SA3_MODEL_DIR="$HOME/models/stable-audio-3-medium"
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)

cleanup() {
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "[sa3-inpainter] starting backend on :5174..."
cd "$DIR"
uv run python backend/server.py &
BACKEND_PID=$!

echo "[sa3-inpainter] starting frontend on :5173..."
cd "$DIR/webui"
node node_modules/.bin/vite --port 5173 &
FRONTEND_PID=$!

sleep 2
xdg-open http://localhost:5173 2>/dev/null || true

wait
