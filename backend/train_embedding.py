"""Train a textual inversion embedding for SA3.

Given a folder of audio files, learns a small embedding vector (N_tokens, 768)
that captures the sonic character of the audio when used as conditioning.

Usage from the server: called via asyncio.create_subprocess_exec with JSON args.
"""
import argparse
import json
import sys
import os
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

SR = 44100
DOWNSAMPLE = 4096


def load_audio_files(folder, max_files=50, max_duration=None):
    """Load audio files from folder, return list of (2, T) float32 numpy arrays."""
    folder = Path(folder)
    exts = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)[:max_files]
    if not files:
        raise ValueError(f"No audio files found in {folder}")

    audios = []
    for f in files:
        try:
            audio, sr = sf.read(f)
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)
            if sr != SR:
                import torchaudio
                a = torch.from_numpy(audio.T).float()
                a = torchaudio.transforms.Resample(sr, SR)(a)
                audio = a.numpy().T
            if max_duration is not None:
                max_samples = int(max_duration * SR)
                if audio.shape[0] > max_samples:
                    audio = audio[:max_samples]
            audios.append(audio.T.astype(np.float32))
        except Exception as e:
            print(f"[train] skip {f.name}: {e}", file=sys.stderr)
    return audios, [f.name for f in files[:len(audios)]]


def encode_audio(pretransform, audio_np, device, use_fp16):
    """Encode (2, T) audio to latent space."""
    audio_t = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    if use_fp16:
        audio_t = audio_t.half()
    with torch.no_grad():
        latent = pretransform.encode(audio_t)
    return latent


def train_embedding(
    model_path,
    audio_folder,
    output_path,
    n_tokens=4,
    steps=500,
    lr=0.005,
    batch_size=1,
    device="cuda",
    use_fp16=True,
    progress_file=None,
    checkpoint_dir=None,
    checkpoint_every=50,
):
    from stable_audio_3.factory import create_diffusion_cond_from_config
    from safetensors.torch import load_file, save_file

    # load model
    print(json.dumps({"status": "loading_model"}), flush=True)
    cfg = json.load(open(f"{model_path}/model_config.json"))
    for c in cfg["model"]["conditioning"]["configs"]:
        if c["type"] == "t5gemma":
            c["config"]["repo_id"] = model_path
    model = create_diffusion_cond_from_config(cfg)
    model.load_state_dict(load_file(f"{model_path}/model.safetensors"), strict=False)
    model.eval().requires_grad_(False).to(device)
    if use_fp16:
        model.half()

    # load audio
    print(json.dumps({"status": "loading_audio"}), flush=True)
    audios, filenames = load_audio_files(audio_folder)
    print(json.dumps({"status": "encoding", "files": len(audios)}), flush=True)

    # encode all audio to latents
    latents = []
    for audio_np in audios:
        lat = encode_audio(model.pretransform, audio_np, device, use_fp16)
        latents.append(lat)
    print(json.dumps({"status": "encoded", "latents": len(latents)}), flush=True)

    # create learnable embedding — initialize at T5Gemma output scale (~100 norm)
    cond_dim = 768  # T5Gemma output dim for SA3
    embedding = torch.randn(n_tokens, cond_dim, device=device, dtype=torch.float32)
    with torch.no_grad():
        embedding.div_(embedding.norm(dim=-1, keepdim=True)).mul_(100.0)
    embedding = torch.nn.Parameter(embedding)
    optimizer = torch.optim.AdamW([embedding], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    # get conditioning keys from model
    cross_attn_cond_ids = model.cross_attn_cond_ids
    global_cond_ids = model.global_cond_ids

    # auto batch size based on free VRAM
    max_lat = 128
    if batch_size <= 0 and device == "cuda":
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        # ~1.5GB per batch element for 128-frame DIT forward+backward
        auto = max(1, int(free_gb / 1.5))
        actual_batch = min(auto, len(latents), 8)
        print(json.dumps({"status": "auto_batch", "free_gb": round(free_gb, 1), "batch_size": actual_batch}), flush=True)
    else:
        actual_batch = min(batch_size if batch_size > 0 else 1, len(latents))
    print(json.dumps({"status": "training", "steps": steps, "batch_size": actual_batch}), flush=True)
    losses = []
    for step in range(steps):
        optimizer.zero_grad()

        # sample a batch of random latents with random crops
        indices = torch.randint(0, len(latents), (actual_batch,))
        x0_list = []
        for idx in indices:
            x0_i = latents[idx.item()]
            if use_fp16:
                x0_i = x0_i.half()
            if x0_i.shape[-1] > max_lat:
                start = torch.randint(0, x0_i.shape[-1] - max_lat, (1,)).item()
                x0_i = x0_i[:, :, start:start + max_lat]
            else:
                # pad shorter latents to max_lat
                pad = max_lat - x0_i.shape[-1]
                x0_i = F.pad(x0_i, (0, pad))
            x0_list.append(x0_i)
        x0 = torch.cat(x0_list, dim=0)  # (B, C, max_lat)

        # sample timesteps per batch element
        t = torch.rand(actual_batch, device=device)

        # noise
        noise = torch.randn_like(x0)
        t_expand = t.view(-1, 1, 1)
        x_t = (1.0 - t_expand) * x0 + t_expand * noise

        # build conditioning with our embedding — repeat for batch
        emb = embedding.to(x0.dtype).unsqueeze(0).expand(actual_batch, -1, -1)
        mask = torch.ones(actual_batch, n_tokens, device=device, dtype=torch.bool)

        # get duration conditioning
        duration_secs = [max_lat * DOWNSAMPLE / SR] * actual_batch
        duration_cond = model.conditioner.conditioners["seconds_total"](duration_secs, device=device)

        # dummy inpaint conditioning (zeros = no inpaint region)
        inpaint_mask = torch.zeros(actual_batch, 1, max_lat, device=device, dtype=x0.dtype)
        inpaint_masked_input = torch.zeros_like(x0)

        cond_tensors = {
            "prompt": (emb, mask),
            "seconds_total": duration_cond,
            "inpaint_mask": (inpaint_mask, None),
            "inpaint_masked_input": (inpaint_masked_input, None),
        }
        cond_inputs = model.get_conditioning_inputs(cond_tensors)

        # forward pass
        with torch.amp.autocast("cuda", enabled=use_fp16):
            v = model.model(x_t, t, **cond_inputs)
            denoised = x_t - t_expand * v
            loss = F.mse_loss(denoised, x0)

        loss.float().backward()
        torch.nn.utils.clip_grad_norm_([embedding], 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)
        avg = np.mean(losses[-25:])
        if step % 25 == 0 or step == steps - 1:
            print(json.dumps({"status": "step", "step": step, "loss": round(avg, 6), "lr": round(scheduler.get_last_lr()[0], 6)}), flush=True)

        # save checkpoint
        if checkpoint_dir and checkpoint_every > 0 and (
            (step > 0 and step % checkpoint_every == 0) or step == steps - 1
        ):
            ckpt_dir = Path(checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"step_{step:05d}_loss_{avg:.6f}.safetensors"
            save_file({"embedding": embedding.data.cpu().float()}, str(ckpt_path))

    # save final embedding as active
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file({"embedding": embedding.data.cpu().float()}, output_path)
    n_ckpts = len(list(Path(checkpoint_dir).glob("*.safetensors"))) if checkpoint_dir else 0
    print(json.dumps({
        "status": "done",
        "output": output_path,
        "tokens": n_tokens,
        "dim": cond_dim,
        "final_loss": round(np.mean(losses[-25:]), 6),
        "files_used": filenames,
        "checkpoints": n_ckpts,
    }), flush=True)

    # cleanup
    del model, latents, embedding, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--audio-folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    train_embedding(
        model_path=args.model_path,
        audio_folder=args.audio_folder,
        output_path=args.output,
        n_tokens=args.tokens,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        use_fp16=args.fp16,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )
