"""Wrapper around SA3's train_lora.py with optional enhancements.

Monkey-patches DiffusionCondTrainingWrapper.__init__ to apply:
- torch.compile on the DiT (SA3_COMPILE=1)
- Gradient checkpointing on transformer blocks (SA3_GRAD_CHECKPOINT=1)
- LoRA on conditioner/T5 encoder (SA3_TRAIN_CONDITIONER=1)

Controlled via env vars since the upstream script doesn't accept these CLI args.
"""
import os
import sys

sa3_root = os.environ.get("SA3_ROOT", os.path.expanduser("~/projects/stable-audio-3"))
if sa3_root not in sys.path:
    sys.path.insert(0, sa3_root)

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper

_orig_init = DiffusionCondTrainingWrapper.__init__

DO_COMPILE = os.environ.get("SA3_COMPILE", "0") == "1"
DO_GRAD_CHECKPOINT = os.environ.get("SA3_GRAD_CHECKPOINT", "0") == "1"
DO_TRAIN_CONDITIONER = os.environ.get("SA3_TRAIN_CONDITIONER", "0") == "1"


def _apply_gradient_checkpointing(dit):
    """Wrap each transformer block's forward with torch.utils.checkpoint."""
    if not hasattr(dit, "transformer") and hasattr(dit, "blocks"):
        blocks = dit.blocks
    elif hasattr(dit, "transformer") and hasattr(dit.transformer, "layers"):
        blocks = dit.transformer.layers
    elif hasattr(dit, "transformer") and hasattr(dit.transformer, "blocks"):
        blocks = dit.transformer.blocks
    else:
        print("[grad_ckpt] Could not find transformer blocks, skipping")
        return

    count = 0
    for block in blocks:
        orig_forward = block.forward

        def make_ckpt_forward(fn):
            def ckpt_forward(*args, **kwargs):
                def run(*a):
                    return fn(*a, **kwargs)
                return torch_checkpoint(run, *args, use_reentrant=False)
            return ckpt_forward

        block.forward = make_ckpt_forward(orig_forward)
        count += 1
    print(f"[grad_ckpt] Wrapped {count} transformer blocks with gradient checkpointing")


def _inject_conditioner_lora(wrapper):
    """Inject LoRA into the conditioner's encoder and unfreeze adapter params."""
    if not hasattr(wrapper, 'diffusion') or not hasattr(wrapper.diffusion, 'conditioner'):
        print("[cond_lora] No conditioner found, skipping")
        return

    conditioner = wrapper.diffusion.conditioner

    try:
        from stable_audio_3.models.lora import inject_lora, get_lora_parameters
    except ImportError:
        print("[cond_lora] SA3 lora module not found, skipping conditioner LoRA")
        return

    rank = int(os.environ.get("SA3_LORA_RANK", "16"))
    adapter_type = os.environ.get("SA3_ADAPTER_TYPE", "dora-rows")
    alpha = float(os.environ.get("SA3_LORA_ALPHA", str(rank)))

    n_injected = 0
    for key, cond in conditioner.conditioners.items():
        if hasattr(cond, 'encoder') or hasattr(cond, 'model'):
            target = getattr(cond, 'encoder', None) or getattr(cond, 'model', None)
            if target is not None:
                n = inject_lora(target, rank=rank, alpha=alpha, adapter_type=adapter_type)
                n_injected += n
                print(f"[cond_lora] Injected {n} adapters into conditioner '{key}'")

    if n_injected > 0:
        conditioner.requires_grad_(False)
        lora_params = get_lora_parameters(conditioner)
        for p in lora_params:
            p.requires_grad_(True)
        conditioner.train()
        print(f"[cond_lora] Total conditioner adapters: {n_injected}, "
              f"trainable params: {sum(p.numel() for p in lora_params)}")

        existing_optim = wrapper.optimizer_configs.get("diffusion", {})
        if existing_optim:
            wrapper.optimizer_configs["conditioner"] = {
                "optimizer": existing_optim.get("optimizer", {
                    "type": "AdamW",
                    "config": {"lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.95]},
                })
            }


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)

    dit = None
    if hasattr(self, 'model') and hasattr(self.model, 'model'):
        dit = self.model.model
    elif hasattr(self, 'diffusion') and hasattr(self.diffusion, 'model'):
        dit = self.diffusion.model

    if DO_GRAD_CHECKPOINT and dit is not None:
        _apply_gradient_checkpointing(dit)

    if DO_COMPILE and dit is not None:
        print("[compile] Compiling DIT with torch.compile(mode='default')...")
        compiled = torch.compile(dit, mode="default")
        if hasattr(self, 'model') and hasattr(self.model, 'model'):
            self.model.model = compiled
        elif hasattr(self, 'diffusion') and hasattr(self.diffusion, 'model'):
            self.diffusion.model = compiled
        print("[compile] DIT compiled")

    if DO_TRAIN_CONDITIONER:
        _inject_conditioner_lora(self)


DiffusionCondTrainingWrapper.__init__ = _patched_init

exec(open(os.path.join(sa3_root, "scripts", "train_lora.py")).read())
