# sa3-inpainter-ui

A browser-based **audio inpainter** for [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) medium, with an MLX-backed SAME-L decoder (no flash-attn needed) and a Svelte UI built around painting on the spectrogram.

![interface](interface.png)

---

## What it does

- **Text-to-audio** — type a prompt, set a length, generate.
- **Audio-to-audio (Vary)** — load a sample, drag the A2A slider, re-render.
- **Inpaint** — paint on the spectrogram, regenerate only those latents; the rest is preserved bit-exact via a stitch with a 256-sample crossfade.

Single track at a time. Modeless interaction — what happens on click/drag depends on where the cursor is. Pinch / scroll zooms (anchored at cursor), shift-scroll pans, click-and-drag paints (shift erases), single-click on the canvas seeks.

Playback runs through a Web Audio graph: the masked regions get a hard-cut lowpass + −2 dB duck while playing, so you can audibly hear what's marked for regeneration.

## Stack

- **Backend** — FastAPI + sa3 medium on torch+MPS for the DIT, MLX for the SAME-L decoder. Loads once (~30 s), serves a small JSON API + audio/png assets.
- **Frontend** — Svelte 5 + Vite, vanilla CSS. Canvas-rendered waveform with one bar per latent. No component library.

## Running

```bash
# backend (loads model at import; sit through ~30s)
uv run python backend/server.py    # binds 127.0.0.1:5174

# frontend (Vite, proxies /api → :5174)
cd webui && npm install && npm run dev
```

Open `http://localhost:5173`.

LoRA library is read from `$SA3_LORA_DIR` (default `~/loras`).

## Layout

```
backend/server.py      FastAPI app, model lifecycle, viz rendering
mlx_sa3/               MLX port of the SA3 medium SAME-L decoder
  ae.py                top-level decoder chain (bottleneck → SAMEDecoder → PatchedPretransform)
  nn_blocks.py         transformer + differential-attention + band-mask SWA
  weights.py           safetensors → mlx weight remap (incl. WNConv1d fold)
webui/                 Svelte 5 frontend (Vite)
  src/lib/             one file per major UI surface
  src/lib/session.svelte.js   shared reactive state + api client
scratch/               correctness tests + dev scripts
design.md              design spec
```

## License

Personal project.
