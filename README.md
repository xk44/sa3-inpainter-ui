# sa3-inpainter-ui

Browser UI for [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) medium inpainting / vary / text-to-audio. MLX-backed SAME-L decoder so it runs on Apple Silicon without flash-attn.

![interface](interface.png)

## Install

```bash
# python deps
uv sync

# frontend deps
cd webui && npm install && cd ..
```

You'll also need the SA3 medium weights from HuggingFace at `~/Projects/stable-audio-3/models/stable-audio-3-medium/` (or edit the `LOCAL_MEDIUM` path in `backend/server.py`).

## Run

```bash
# backend on :5174 — ~30s to load the model
uv run python backend/server.py

# frontend on :5173 — Vite proxies /api → :5174
cd webui && npm run dev
```

Open http://localhost:5173.

LoRA library is read from `$SA3_LORA_DIR` (default `~/loras`).
