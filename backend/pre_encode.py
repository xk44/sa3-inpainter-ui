"""Pre-encode audio dataset to latents for faster LoRA training.

Calls SA3's pre_encode_dataset.py to encode audio + captions into .npy/.json
pairs that skip the VAE + conditioner during training.

Usage from server: spawned via asyncio.create_subprocess_exec.
"""
import argparse
import json
import sys
import os
import shutil
from pathlib import Path


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
        txt_path = dst.with_suffix(".txt")
        txt_path.write_text(caption)
        prepared.append(f.name)

    return prepared


def pre_encode(audio_folder, output_dir, caption, ae_model="same-l"):
    import subprocess

    work_dir = Path(output_dir) / "_dataset"
    print(json.dumps({"status": "preparing_dataset"}), flush=True)
    files = prepare_dataset(audio_folder, work_dir, caption)
    print(json.dumps({"status": "dataset_ready", "files": len(files)}), flush=True)

    sa3_root = Path(os.environ.get("SA3_ROOT", str(Path.home() / "projects/stable-audio-3")))
    encode_script = sa3_root / "scripts" / "pre_encode_dataset.py"
    if not encode_script.exists():
        raise FileNotFoundError(f"pre_encode_dataset.py not found at {encode_script}")

    encoded_dir = str(Path(output_dir) / "_encoded")
    Path(encoded_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(encode_script),
        "--model", ae_model,
        "--data_dir", str(work_dir),
        "--output_path", encoded_dir,
        "--batch_size", "1",
        "--model_half",
    ]

    print(json.dumps({"status": "encoding", "cmd": " ".join(cmd)}), flush=True)

    env = os.environ.copy()
    pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(sa3_root) + (":" + pypath if pypath else "")
    if not env.get("HF_TOKEN"):
        token_file = Path.home() / ".cache/huggingface/token"
        if token_file.exists():
            env["HF_TOKEN"] = token_file.read_text().strip()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Processing batch"):
            try:
                batch_num = int(line.split()[-1])
                print(json.dumps({"status": "encoding_batch", "batch": batch_num, "total": len(files)}), flush=True)
            except (ValueError, IndexError):
                pass
        elif line.startswith("{"):
            print(line, flush=True)
        else:
            print(json.dumps({"status": "log", "msg": line[:200]}), flush=True)

    proc.wait()

    if proc.returncode != 0:
        print(json.dumps({"status": "error", "returncode": proc.returncode}), flush=True)
        sys.exit(1)

    n_latents = len(list(Path(encoded_dir).glob("*.npy")))
    print(json.dumps({
        "status": "done",
        "encoded_dir": encoded_dir,
        "latents": n_latents,
        "files": len(files),
    }), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audio-folder", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--caption", default="")
    p.add_argument("--ae-model", default="same-l")
    args = p.parse_args()

    pre_encode(
        audio_folder=args.audio_folder,
        output_dir=args.output_dir,
        caption=args.caption,
        ae_model=args.ae_model,
    )
