"""
Cross-attention KV cache for SA3 DiffusionTransformer.

The text conditioning (cross_attn_cond) is identical for every diffusion step.
This module caches the fully-processed K/V tensors (post to_kv, post QK-norm,
post RoPE) from the first diffusion step and reuses them in subsequent steps,
skipping the redundant projection work.

Usage:
    from kv_cache import enable_kv_cache, disable_kv_cache, clear_kv_cache

    # Before generation:
    enable_kv_cache(sa.model.model)   # DiffusionTransformer

    # After generation (always in finally):
    clear_kv_cache(sa.model.model)
    disable_kv_cache(sa.model.model)

Architecture:
    DiffusionTransformer
      └─ transformer (ContinuousTransformer)
           └─ layers[i] (TransformerBlock)
                └─ cross_attn (Attention) — has to_q + to_kv (cross-attention)

    Attention.forward(x, context=...) flow:
        q = to_q(x)
        k, v = to_kv(context).chunk(2)
        [qk_norm, RoPE on q and k]
        out = apply_attn(q, k, v)
        out = to_out(out)

    We patch apply_attn on cross_attn instances. On the first call, we capture
    k and v (already fully processed). On subsequent calls we skip recomputing
    them by injecting the cached values. But since we can't easily intercept
    before apply_attn is called without touching the full forward path, the
    cleaner design is:

    Patch Attention.forward on each cross_attn:
    - On first call (context seen for the first time by shape/device key):
        run orig_forward normally, but additionally capture k,v by wrapping
        apply_attn to sniff its inputs.
    - On subsequent calls:
        compute only q = to_q(x) + qk_norm + RoPE, reuse cached k,v,
        call apply_attn(q, cached_k, cached_v), then to_out.

    Cache key: (batch_size, context_seq_len, device) so CFG-doubled batches
    cache independently from single batches.

Note on RoPE: SA3 sets cross_attn_rotary_pos_emb=False, so rotary_pos_emb is
None for cross-attention in practice. The code handles it correctly either way.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

_ORIG_FORWARD_ATTR = "_kvc_orig_forward"
_CACHE_ATTR        = "_kvc_store"       # {cache_key: (k, v)} on each Attention
_ENABLED_ATTR      = "_kvc_enabled"


def _make_cached_forward(attn):
    """Return a replacement Attention.forward that caches K/V on first call."""

    orig_forward = getattr(attn, _ORIG_FORWARD_ATTR)

    def cached_forward(
        x,
        context=None,
        rotary_pos_emb=None,
        rotary_pos_emb_k=None,
        causal=None,
        flex_attention_block_mask=None,
        flex_attention_score_mod=None,
        flash_attn_sliding_window=None,
        padding_mask=None,
        varlen_metadata=None,
    ):
        # Only cache cross-attention (context present, has separate to_q/to_kv, non-differential)
        if (
            context is None
            or not hasattr(attn, "to_kv")
            or attn.differential
        ):
            return orig_forward(
                x, context=context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_emb_k=rotary_pos_emb_k,
                causal=causal,
                flex_attention_block_mask=flex_attention_block_mask,
                flex_attention_score_mod=flex_attention_score_mod,
                flash_attn_sliding_window=flash_attn_sliding_window,
                padding_mask=padding_mask,
                varlen_metadata=varlen_metadata,
            )

        cache_key = (context.shape[0], context.shape[1], str(context.device))
        store: dict = getattr(attn, _CACHE_ATTR)

        if cache_key in store:
            # Cache hit: compute Q only, reuse cached K, V
            cached_k, cached_v = store[cache_key]
            return _forward_with_cached_kv(
                attn, x, cached_k, cached_v,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_emb_k=rotary_pos_emb_k,
                causal=causal,
                flex_attention_block_mask=flex_attention_block_mask,
                flex_attention_score_mod=flex_attention_score_mod,
                flash_attn_sliding_window=flash_attn_sliding_window,
                padding_mask=padding_mask,
                varlen_metadata=varlen_metadata,
            )

        # Cache miss: run full forward, sniff K/V inside apply_attn
        _sniffed = {}
        orig_apply_attn = attn.apply_attn

        def _sniffing_apply_attn(q, k, v, **kw):
            _sniffed["k"] = k.detach().clone()
            _sniffed["v"] = v.detach().clone()
            return orig_apply_attn(q, k, v, **kw)

        attn.apply_attn = _sniffing_apply_attn
        try:
            result = orig_forward(
                x, context=context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_emb_k=rotary_pos_emb_k,
                causal=causal,
                flex_attention_block_mask=flex_attention_block_mask,
                flex_attention_score_mod=flex_attention_score_mod,
                flash_attn_sliding_window=flash_attn_sliding_window,
                padding_mask=padding_mask,
                varlen_metadata=varlen_metadata,
            )
        finally:
            attn.apply_attn = orig_apply_attn

        if "k" in _sniffed and "v" in _sniffed:
            store[cache_key] = (_sniffed["k"], _sniffed["v"])

        return result

    return cached_forward


def _forward_with_cached_kv(
    attn,
    x,
    cached_k,
    cached_v,
    rotary_pos_emb=None,
    rotary_pos_emb_k=None,
    causal=None,
    flex_attention_block_mask=None,
    flex_attention_score_mod=None,
    flash_attn_sliding_window=None,
    padding_mask=None,
    varlen_metadata=None,
):
    """Run attention with pre-computed K/V. Only Q is projected from x."""
    h = attn.num_heads

    # Project Q
    q = attn.to_q(x)
    q = rearrange(q, "b n (h d) -> b h n d", h=h)

    # QK norm on Q
    if attn.qk_norm == "l2":
        q = F.normalize(q, dim=-1, eps=attn.qk_norm_eps)
    elif attn.qk_norm != "none":
        q_type = q.dtype
        q = attn.q_norm(q).to(q_type)

    # RoPE on Q (K already has RoPE baked in from first call)
    if rotary_pos_emb is not None:
        from stable_audio_3.models.transformer import apply_rotary_pos_emb
        freqs, _ = rotary_pos_emb
        q_dtype = q.dtype
        q = q.to(torch.float32)
        freqs = freqs.to(torch.float32)

        # Match the ratio logic from Attention.forward
        k_seq = cached_k.shape[-2]
        q_seq = q.shape[-2]
        if q_seq >= k_seq and k_seq > 0:
            ratio = q_seq / k_seq
            q_freqs = ratio * freqs
        else:
            q_freqs = freqs

        q = apply_rotary_pos_emb(q, q_freqs)
        q = q.to(cached_v.dtype)

    k = cached_k
    v = cached_v

    causal_flag = attn.causal if causal is None else causal
    if q.shape[-2] == 1 and causal_flag:
        causal_flag = False

    out = attn.apply_attn(
        q, k, v,
        causal=causal_flag,
        flex_attention_block_mask=flex_attention_block_mask,
        flex_attention_score_mod=flex_attention_score_mod,
        flash_attn_sliding_window=flash_attn_sliding_window,
        padding_mask=padding_mask,
        varlen_metadata=varlen_metadata,
    )

    out = rearrange(out, "b h n d -> b n (h d)")
    out = attn.to_out(out)

    if attn.feat_scale:
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).to(out.dtype)
            out_dc = (out * mask).sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True).clamp(min=1)
            out_hf = out - out_dc
            out = out + (attn.lambda_dc * out_dc + attn.lambda_hf * out_hf) * mask
        else:
            out_dc = out.mean(dim=-2, keepdim=True)
            out_hf = out - out_dc
            out = out + attn.lambda_dc * out_dc + attn.lambda_hf * out_hf

    return out


def _get_cross_attn_modules(dit_model):
    """Yield all cross_attn Attention instances from a DiffusionTransformer."""
    try:
        transformer = dit_model.transformer
    except AttributeError:
        return
    for layer in transformer.layers:
        if getattr(layer, "cross_attend", False) and hasattr(layer, "cross_attn"):
            yield layer.cross_attn


def enable_kv_cache(dit_model):
    """
    Patch all cross-attention layers in dit_model to cache K/V projections.
    Idempotent — safe to call multiple times.
    """
    count = 0
    for attn in _get_cross_attn_modules(dit_model):
        if getattr(attn, _ENABLED_ATTR, False):
            continue
        setattr(attn, _ORIG_FORWARD_ATTR, attn.forward)
        setattr(attn, _CACHE_ATTR, {})
        setattr(attn, _ENABLED_ATTR, True)
        attn.forward = _make_cached_forward(attn)
        count += 1
    print(f"[kv_cache] enabled on {count} cross-attn layers")


def disable_kv_cache(dit_model):
    """Restore original Attention.forward() on all patched cross-attn layers."""
    count = 0
    for attn in _get_cross_attn_modules(dit_model):
        if not getattr(attn, _ENABLED_ATTR, False):
            continue
        orig = getattr(attn, _ORIG_FORWARD_ATTR, None)
        if orig is not None:
            attn.forward = orig
        setattr(attn, _ENABLED_ATTR, False)
        count += 1
    if count:
        print(f"[kv_cache] disabled on {count} cross-attn layers")


def clear_kv_cache(dit_model):
    """Clear cached K/V tensors. Call between generation runs."""
    total = 0
    for attn in _get_cross_attn_modules(dit_model):
        store = getattr(attn, _CACHE_ATTR, None)
        if store:
            total += len(store)
            store.clear()
    print(f"[kv_cache] cleared ({total} cached entries)")
