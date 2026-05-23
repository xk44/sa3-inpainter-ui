"""LoRA training wrapper for SA3 inpainter UI.

Prepares a dataset folder (audio + .txt captions), then calls SA3's
built-in train_lora.py script. Reports progress via JSON lines on stdout.

Usage from server: spawned via asyncio.create_subprocess_exec with JSON args.
"""
import argparse
import json
import sys
import os
import gc
import shutil
import time
from pathlib import Path

import soundfile as sf


def prepare_dataset(audio_folder, work_dir, caption, max_files=50):
    """Copy audio files to work_dir with matching .txt caption files."""
    audio_folder = Path(audio_folder)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    exts = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}
    files = sorted(p for p in audio_folder.iterdir() if p.suffix.lower() in exts)[:max_files]
    if not files:
        raise ValueError(f"No audio files found in {audio_folder}")

    prepared = []
    for f in files:
        dst = work_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
        # write caption
        txt_path = dst.with_suffix(".txt")
        txt_path.write_text(caption)
        prepared.append(f.name)

    return prepared


def train_lora(
    model_name,
    audio_folder,
    output_dir,
    caption="",
    rank=16,
    adapter_type="dora-rows",
    steps=1000,
    lr=1e-4,
    batch_size=1,
    checkpoint_every=100,
    base_precision="bf16",
    exclude=None,
    encoded_dir=None,
    use_compile=False,
    grad_checkpoint=False,
    train_conditioner=False,
):
    import subprocess

    sa3_root = Path(os.environ.get("SA3_ROOT", str(Path.home() / "projects/stable-audio-3")))
    train_script = sa3_root / "scripts" / "train_lora.py"
    if not train_script.exists():
        raise FileNotFoundError(f"SA3 train_lora.py not found at {train_script}")

    # use pre-encoded latents if available, otherwise prepare raw dataset
    use_encoded = False
    enc_path = Path(output_dir) / "_encoded"
    if encoded_dir and Path(encoded_dir).is_dir() and list(Path(encoded_dir).glob("*.npy")):
        use_encoded = True
        enc_path = Path(encoded_dir)
        print(json.dumps({"status": "using_preencoded", "dir": str(enc_path), "latents": len(list(enc_path.glob("*.npy")))}), flush=True)
    elif enc_path.is_dir() and list(enc_path.glob("*.npy")):
        use_encoded = True
        print(json.dumps({"status": "using_preencoded", "dir": str(enc_path), "latents": len(list(enc_path.glob("*.npy")))}), flush=True)
    else:
        work_dir = Path(output_dir) / "_dataset"
        print(json.dumps({"status": "preparing_dataset"}), flush=True)
        files = prepare_dataset(audio_folder, work_dir, caption)
        print(json.dumps({"status": "dataset_ready", "files": len(files)}), flush=True)

    save_dir = str(Path(output_dir) / "checkpoints")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    use_enhanced = use_compile or grad_checkpoint or train_conditioner
    if use_enhanced:
        cmd_script = str(Path(__file__).resolve().parent / "train_lora_compiled.py")
    else:
        cmd_script = str(train_script)

    cmd = [
        sys.executable, cmd_script,
        "--model", model_name,
        "--save_dir", save_dir,
        "--rank", str(rank),
        "--adapter_type", adapter_type,
        "--steps", str(steps),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--checkpoint_every", str(checkpoint_every),
        "--base_precision", base_precision,
        "--logger", "csv",
        "--num_workers", "2",
        "--demo_every", str(max(steps, 99999)),
    ]
    if use_encoded:
        cmd.extend(["--encoded_dir", str(enc_path)])
    else:
        cmd.extend(["--data_dir", str(work_dir)])
    if exclude:
        cmd.extend(["--exclude"] + exclude)

    print(json.dumps({"status": "training", "steps": steps, "cmd": " ".join(cmd)}), flush=True)

    # run training, stream output — ensure SA3 is on PYTHONPATH
    env = os.environ.copy()
    pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(sa3_root) + (":" + pypath if pypath else "")
    # pass HF token if available (needed for gated model access)
    if not env.get("HF_TOKEN"):
        token_file = Path.home() / ".cache/huggingface/token"
        if token_file.exists():
            env["HF_TOKEN"] = token_file.read_text().strip()
    # enhanced wrapper env vars
    if use_compile:
        env["SA3_COMPILE"] = "1"
    if grad_checkpoint:
        env["SA3_GRAD_CHECKPOINT"] = "1"
    if train_conditioner:
        env["SA3_TRAIN_CONDITIONER"] = "1"
        env["SA3_LORA_RANK"] = str(rank)
        env["SA3_ADAPTER_TYPE"] = adapter_type
        env["SA3_LORA_ALPHA"] = str(rank)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    last_step = 0
    last_loss = 0.0
    last_epoch = 0
    steps_per_epoch = None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        # PL logs like: Epoch 2: 60%|██████ | 3/5 [00:08<00:05, 1.55it/s, v_num=0, train/loss=0.123]
        if "train/loss=" in line:
            try:
                loss_str = line.split("train/loss=")[-1].rstrip("]").split(",")[0]
                last_loss = float(loss_str)
            except (ValueError, IndexError):
                pass
            # extract epoch number
            try:
                epoch_str = line.split("Epoch ")[1].split(":")[0].strip()
                last_epoch = int(epoch_str)
            except (ValueError, IndexError):
                pass
            # extract epoch_step/steps_per_epoch and compute global step
            try:
                progress_part = line.split("|")[-1].strip()
                frac = progress_part.split("[")[0].strip()
                epoch_step, total = frac.split("/")
                epoch_step = int(epoch_step.strip())
                total = int(total.strip())
                if steps_per_epoch is None:
                    steps_per_epoch = total
                last_step = last_epoch * steps_per_epoch + epoch_step
            except (ValueError, IndexError):
                pass
            print(json.dumps({"status": "step", "step": last_step, "loss": round(last_loss, 6)}), flush=True)
        elif line.startswith("{"):
            print(line, flush=True)
        else:
            print(json.dumps({"status": "log", "msg": line[:200]}), flush=True)

    proc.wait()

    if proc.returncode != 0:
        print(json.dumps({"status": "error", "returncode": proc.returncode}), flush=True)
        sys.exit(1)

    # find the final checkpoint
    ckpt_dir = Path(save_dir)
    ckpts = sorted(ckpt_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)

    # convert PL checkpoints to safetensors
    final_path = None
    if ckpts:
        print(json.dumps({"status": "converting_checkpoints", "count": len(ckpts)}), flush=True)
        import torch
        from safetensors.torch import save_file

        for ckpt_path in ckpts:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state_dict = ckpt.get("state_dict", ckpt)
                # extract only LoRA keys
                lora_sd = {k: v for k, v in state_dict.items()
                          if "lora_" in k or "magnitude" in k or "M_xs" in k}
                if not lora_sd:
                    # try removing "model." prefix
                    lora_sd = {k.replace("model.", "", 1): v for k, v in state_dict.items()
                              if "lora_" in k or "magnitude" in k or "M_xs" in k}

                if lora_sd:
                    st_path = ckpt_path.with_suffix(".safetensors")
                    lora_config = {
                        "rank": rank,
                        "alpha": rank,
                        "adapter_type": adapter_type,
                    }
                    if exclude:
                        lora_config["exclude"] = exclude
                    metadata = {"lora_config": json.dumps(lora_config)}
                    fp16_dict = {k: v.half() if v.is_floating_point() else v for k, v in lora_sd.items()}
                    save_file(fp16_dict, str(st_path), metadata=metadata)
                    final_path = str(st_path)
                    print(json.dumps({"status": "checkpoint_converted", "path": str(st_path), "keys": len(lora_sd)}), flush=True)
            except Exception as e:
                print(json.dumps({"status": "convert_error", "file": str(ckpt_path), "error": str(e)}), flush=True)

    # copy final checkpoint to LORA_DIR for easy loading
    lora_dir = Path(os.environ.get("SA3_LORA_DIR", str(Path.home() / "loras")))
    lora_dir.mkdir(parents=True, exist_ok=True)
    lora_name = Path(output_dir).name
    if final_path:
        dst = lora_dir / f"{lora_name}.safetensors"
        shutil.copy2(final_path, dst)
        print(json.dumps({
            "status": "done",
            "output": str(dst),
            "name": lora_name,
            "rank": rank,
            "adapter_type": adapter_type,
            "steps": steps,
        }), flush=True)
    else:
        print(json.dumps({"status": "done", "output": None, "warning": "no checkpoints found"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="medium-base")
    p.add_argument("--audio-folder", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--caption", default="")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--adapter-type", default="dora-rows")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--base-precision", default="bf16")
    p.add_argument("--exclude", nargs="*", default=None)
    p.add_argument("--encoded-dir", default=None)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--grad-checkpoint", action="store_true", default=False)
    p.add_argument("--train-conditioner", action="store_true", default=False)
    args = p.parse_args()

    train_lora(
        model_name=args.model_name,
        audio_folder=args.audio_folder,
        output_dir=args.output_dir,
        caption=args.caption,
        rank=args.rank,
        adapter_type=args.adapter_type,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        base_precision=args.base_precision,
        exclude=args.exclude,
        encoded_dir=args.encoded_dir,
        use_compile=args.compile,
        grad_checkpoint=args.grad_checkpoint,
        train_conditioner=args.train_conditioner,
    )
