"""
Token Merging (ToMe) for SA3 DiffusionTransformer.

Reduces the self-attention sequence length by merging similar latent tokens
before each transformer block's self-attention, then unmerging after the
full block computation. This is a clean monkey-patch — no SA3 source changes.

Reference: Bolya et al. "Token Merging: Your ViT But Faster" (ICLR 2023)
           https://arxiv.org/abs/2210.09461

Usage:
    from tome import apply_tome, remove_tome

    apply_tome(sa.model.model, ratio=0.25)   # merge 25% of tokens per block
    # ... generate ...
    remove_tome(sa.model.model)

Architecture:
    - DiffusionTransformer.transformer → ContinuousTransformer
    - ContinuousTransformer.layers[i]  → TransformerBlock
    - TransformerBlock.forward(x, ...)  — we wrap this to inject merge/unmerge
      around the full block (self-attn + cross-attn + ff).

Bipartite soft matching:
    Tokens are split into two sets A (even indices) and B (odd indices).
    For each token in A, find its most similar token in B via cosine similarity.
    The top-r pairs are merged (averaged). After the block, unmerge by
    scattering the merged output back to the original positions.

Memory tokens (prepended at the start of x inside ContinuousTransformer)
are excluded from merging — only the audio latent tokens are merged.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Callable


# Attribute names used to tag patched modules
_TOME_ORIG_FORWARD = "_tome_orig_forward"
_TOME_RATIO = "_tome_ratio"
_TOME_ENABLED = "_tome_enabled"


# ---------------------------------------------------------------------------
# Bipartite soft matching
# ---------------------------------------------------------------------------

def bipartite_soft_matching(
    x: torch.Tensor,
    r: int,
) -> Tuple[Callable, Callable]:
    """
    Compute merge/unmerge functions for token sequence x using bipartite matching.

    Args:
        x: (B, N, D) token sequence
        r: number of token pairs to merge (r tokens are eliminated, so output
           has N - r tokens instead of N)

    Returns:
        merge_fn:   (B, N, D) -> (B, N-r, D)
        unmerge_fn: (B, N-r, D) -> (B, N, D)
    """
    B, N, D = x.shape

    if r <= 0 or N <= 2:
        identity = lambda t: t
        return identity, identity

    # Split into two sets: A = even indices, B = odd indices
    n_a = (N + 1) // 2
    n_b = N // 2

    x_a = x[:, ::2, :]   # (B, n_a, D)
    x_b = x[:, 1::2, :]  # (B, n_b, D)

    # Cosine similarity matrix: (B, n_a, n_b)
    a_norm = F.normalize(x_a.float(), dim=-1)
    b_norm = F.normalize(x_b.float(), dim=-1)
    sim = torch.bmm(a_norm, b_norm.transpose(1, 2))  # (B, n_a, n_b)

    # For each a, find best matching b
    scores, best_b = sim.max(dim=-1)  # (B, n_a)

    # Select top-r pairs by similarity score
    r_actual = min(r, n_a, n_b)
    _, top_r_idx = scores.topk(r_actual, dim=-1)  # (B, r_actual)

    # Build merge mapping:
    #   merged[i] = mean(a[top_r_idx[i]], b[best_b[top_r_idx[i]]])
    # Unmerged a tokens stay in place; all b tokens are kept except those merged.

    device = x.device
    dtype = x.dtype

    # Indices of a tokens that get merged
    a_merge_idx = top_r_idx  # (B, r_actual)  — indices into x_a
    # Indices of b tokens they merge with
    b_merge_idx = best_b.gather(1, a_merge_idx)  # (B, r_actual)

    # Build a boolean mask for which a tokens are merged
    a_merged_mask = torch.zeros(B, n_a, dtype=torch.bool, device=device)
    a_merged_mask.scatter_(1, a_merge_idx, True)

    # Build a boolean mask for which b tokens are merged
    b_merged_mask = torch.zeros(B, n_b, dtype=torch.bool, device=device)
    b_merged_mask.scatter_(1, b_merge_idx, True)

    # Unmerged a indices (in original sequence: position 2*i)
    # Unmerged b indices (in original sequence: position 2*j+1)

    def merge(tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N - r_actual, D)"""
        B2, N2, D2 = tokens.shape
        t_a = tokens[:, ::2, :]    # (B, n_a, D)
        t_b = tokens[:, 1::2, :]   # (B, n_b, D)

        # Merged tokens: average of paired a and b
        # Gather b tokens for each merging a
        b_for_merge = t_b.gather(
            1, b_merge_idx.unsqueeze(-1).expand(-1, -1, D2)
        )  # (B, r_actual, D)
        a_for_merge = t_a.gather(
            1, a_merge_idx.unsqueeze(-1).expand(-1, -1, D2)
        )  # (B, r_actual, D)
        merged_tokens = 0.5 * (a_for_merge + b_for_merge)  # (B, r_actual, D)

        # Unmerged a tokens
        a_unmerged_mask_expanded = (~a_merged_mask).unsqueeze(-1).expand_as(t_a)
        unmerged_a = t_a[a_unmerged_mask_expanded].reshape(B2, n_a - r_actual, D2)

        # Unmerged b tokens
        b_unmerged_mask_expanded = (~b_merged_mask).unsqueeze(-1).expand_as(t_b)
        unmerged_b = t_b[b_unmerged_mask_expanded].reshape(B2, n_b - r_actual, D2)

        # Concatenate: [merged | unmerged_a | unmerged_b]
        # We'll track positions via indices stored in closure for unmerge
        result = torch.cat([merged_tokens, unmerged_a, unmerged_b], dim=1)
        return result

    def unmerge(tokens: torch.Tensor) -> torch.Tensor:
        """(B, N - r_actual, D) -> (B, N, D)"""
        B2 = tokens.shape[0]
        D2 = tokens.shape[-1]

        merged_tokens = tokens[:, :r_actual, :]                 # (B, r_actual, D)
        unmerged_a    = tokens[:, r_actual:r_actual + (n_a - r_actual), :]
        unmerged_b    = tokens[:, r_actual + (n_a - r_actual):, :]

        # Reconstruct full t_a: insert merged back at a_merge_idx, unmerged at rest
        t_a_out = torch.zeros(B2, n_a, D2, device=tokens.device, dtype=tokens.dtype)
        # Scatter unmerged a
        unmerged_a_idx = (~a_merged_mask).nonzero(as_tuple=False)  # (B*(n_a-r), 2)
        # Easier: use scatter_ with the inverse mask indices
        a_unmerged_positions = torch.zeros(B2, n_a, dtype=torch.long, device=device)
        # Build sorted positions for unmerged a
        # For each batch item, positions of False in a_merged_mask
        for b in range(B2):
            unmerged_pos = (~a_merged_mask[b]).nonzero(as_tuple=True)[0]  # (n_a - r,)
            if len(unmerged_pos) > 0:
                t_a_out[b].index_copy_(0, unmerged_pos, unmerged_a[b, :len(unmerged_pos)])
            t_a_out[b].index_copy_(0, a_merge_idx[b], merged_tokens[b])

        # Reconstruct full t_b
        t_b_out = torch.zeros(B2, n_b, D2, device=tokens.device, dtype=tokens.dtype)
        for b in range(B2):
            unmerged_pos_b = (~b_merged_mask[b]).nonzero(as_tuple=True)[0]
            if len(unmerged_pos_b) > 0:
                t_b_out[b].index_copy_(0, unmerged_pos_b, unmerged_b[b, :len(unmerged_pos_b)])
            # merged b gets the merged token value
            t_b_out[b].index_copy_(0, b_merge_idx[b], merged_tokens[b])

        # Interleave a and b back to original order
        out = torch.zeros(B2, N, D2, device=tokens.device, dtype=tokens.dtype)
        out[:, ::2, :] = t_a_out
        out[:, 1::2, :] = t_b_out
        return out

    return merge, unmerge


# ---------------------------------------------------------------------------
# TransformerBlock wrapping
# ---------------------------------------------------------------------------

def _make_tome_forward(block, num_memory_tokens: int, ratio: float):
    """
    Return a replacement forward() for TransformerBlock that merges tokens
    before the block and unmerges after.

    Memory tokens occupy the first `num_memory_tokens` positions in x; they
    are excluded from merging.
    """
    orig_forward = getattr(block, _TOME_ORIG_FORWARD)

    def tome_forward(x, **kwargs):
        B, N, D = x.shape

        # Separate memory tokens from audio tokens
        mem = x[:, :num_memory_tokens, :]      # (B, M, D) — may be empty
        audio = x[:, num_memory_tokens:, :]    # (B, T, D)

        T = audio.shape[1]
        r = int(T * ratio)

        if r > 0 and T > 2:
            merge_fn, unmerge_fn = bipartite_soft_matching(audio, r)
            audio_merged = merge_fn(audio)
            x_in = torch.cat([mem, audio_merged], dim=1) if num_memory_tokens > 0 else audio_merged
        else:
            merge_fn = unmerge_fn = None
            x_in = x

        # Run the original block forward
        x_out = orig_forward(x_in, **kwargs)

        # Unmerge
        if unmerge_fn is not None:
            mem_out = x_out[:, :num_memory_tokens, :]
            audio_out = x_out[:, num_memory_tokens:, :]
            audio_out = unmerge_fn(audio_out)
            x_out = torch.cat([mem_out, audio_out], dim=1) if num_memory_tokens > 0 else audio_out

        return x_out

    return tome_forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_tome(dit_model, ratio: float = 0.25):
    """
    Patch all TransformerBlock layers in dit_model to use Token Merging.

    Args:
        dit_model: DiffusionTransformer (sa.model.model)
        ratio: fraction of audio tokens to merge per block (0.0 – 0.5)
               0.25 = merge 25% of tokens → ~25% sequence reduction

    Safe to call multiple times; re-applies with new ratio each call.
    """
    if ratio <= 0:
        return

    try:
        transformer = dit_model.transformer  # ContinuousTransformer
    except AttributeError:
        print("[tome] no transformer found on model, skipping")
        return

    num_memory_tokens = getattr(transformer, "num_memory_tokens", 0)

    for layer in transformer.layers:
        # Remove previous patch if any
        if getattr(layer, _TOME_ENABLED, False):
            orig = getattr(layer, _TOME_ORIG_FORWARD, None)
            if orig is not None:
                layer.forward = orig

        setattr(layer, _TOME_ORIG_FORWARD, layer.forward)
        setattr(layer, _TOME_RATIO, ratio)
        setattr(layer, _TOME_ENABLED, True)
        layer.forward = _make_tome_forward(layer, num_memory_tokens, ratio)

    print(f"[tome] applied ratio={ratio:.2f} across {len(transformer.layers)} blocks "
          f"(memory_tokens={num_memory_tokens})")


def remove_tome(dit_model):
    """Restore original TransformerBlock.forward() on all patched layers."""
    try:
        transformer = dit_model.transformer
    except AttributeError:
        return

    count = 0
    for layer in transformer.layers:
        if not getattr(layer, _TOME_ENABLED, False):
            continue
        orig = getattr(layer, _TOME_ORIG_FORWARD, None)
        if orig is not None:
            layer.forward = orig
        setattr(layer, _TOME_ENABLED, False)
        count += 1

    if count:
        print(f"[tome] removed from {count} blocks")
