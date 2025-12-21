# self_attention_batch.py
# Batched bounded self-attention for StoryBooth-style regional self-attention.
#
# This file is adapted from the user's self_attention.py. It keeps backward
# compatibility with single-image layout, and adds support for per-sample layouts:
#   self_obj.split_ratio can be:
#     - single layout: List[drow]
#     - batched layout: List[List[drow]]  (one layout per sample)
#
# It also supports classifier-free guidance batches where hidden_states batch
# is 2x the prompt batch size by repeating layouts.

import math
import torch
import xformers  # noqa: F401  (kept for parity; we use torch matmul here)
from cross_attention import split_dims


def _is_row_obj(x):
    """Heuristic: a 'row' object has .start/.end and .cols."""
    return hasattr(x, "start") and hasattr(x, "end") and hasattr(x, "cols")


def _normalize_layouts_for_batch(self_obj, bsz):
    """
    Returns a list of layouts with length == bsz.
    Each layout is List[drow].
    """
    if not hasattr(self_obj, "split_ratio") or self_obj.split_ratio is None:
        return [None] * bsz

    sr = self_obj.split_ratio

    # Single layout: List[drow]
    if isinstance(sr, (list, tuple)) and (len(sr) == 0 or _is_row_obj(sr[0])):
        return [sr] * bsz

    # Batched layouts: List[List[drow]]
    if isinstance(sr, (list, tuple)) and len(sr) > 0 and isinstance(sr[0], (list, tuple)):
        layouts = list(sr)
        n = len(layouts)
        if n == 0:
            return [None] * bsz

        # Common cases:
        # - n == bsz: already aligned
        # - bsz == 2*n (CFG): duplicate for cond/uncond
        if n == bsz:
            return layouts
        if bsz == 2 * n:
            return layouts + layouts

        # Best-effort fallback: repeat modulo (avoids crashing in weird schedules)
        out = []
        for i in range(bsz):
            out.append(layouts[i % n])
        return out

    # Unknown structure -> treat as missing
    return [None] * bsz


def _build_region_masks_from_layout(layout, latent_h, latent_w, device):
    """
    Build per-region token masks from a *single* layout (List[drow]).
    Each region mask is a flattened bool tensor of length N = latent_h * latent_w.
    """
    if layout is None:
        return None

    region_masks = []
    for drow in layout:
        row_start = int(latent_h * float(drow.start))
        row_end = int(latent_h * float(drow.end))

        for dcell in drow.cols:
            col_start = int(latent_w * float(dcell.start))
            col_end = int(latent_w * float(dcell.end))

            # safeguard against degenerate / rounding issues
            row_start_c = max(0, min(latent_h, row_start))
            row_end_c = max(0, min(latent_h, row_end))
            col_start_c = max(0, min(latent_w, col_start))
            col_end_c = max(0, min(latent_w, col_end))

            if row_end_c <= row_start_c or col_end_c <= col_start_c:
                continue

            mask = torch.zeros(latent_h, latent_w, dtype=torch.bool, device=device)
            mask[row_start_c:row_end_c, col_start_c:col_end_c] = True
            region_masks.append(mask.view(-1))  # (N,)

    if len(region_masks) == 0:
        return None

    return region_masks


def _vanilla_self_attention(module, hidden_states):
    """
    Plain self-attention with the same math as diffusers' Attention,
    but without any bounding.
    """
    query = module.to_q(hidden_states)
    key = module.to_k(hidden_states)
    value = module.to_v(hidden_states)

    query = module.head_to_batch_dim(query)
    key = module.head_to_batch_dim(key)
    value = module.head_to_batch_dim(value)

    head_dim = query.shape[-1]
    scale = head_dim ** -0.5

    attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    attn_probs = attn_scores.softmax(dim=-1)

    hidden_states = torch.matmul(attn_probs, value)
    hidden_states = module.batch_to_head_dim(hidden_states)

    hidden_states = module.to_out[0](hidden_states)
    hidden_states = module.to_out[1](hidden_states)
    return hidden_states


def _compute_bounded_self_attention(self_obj, module, hidden_states):
    """
    Batched bounded self-attention:
    - For each sample in the batch, build a token-to-token allow mask based on its layout.
    - Add a large negative bias for disallowed pairs (emulating log(M)).
    - Keep dropout relaxation (Nr > beta_d) per sample.
    """
    bsz, n_tokens, _ = hidden_states.shape
    height = getattr(self_obj, "h", None)
    width = getattr(self_obj, "w", None)

    # Fallback: if we don't know spatial size, just do vanilla attention
    if height is None or width is None:
        return _vanilla_self_attention(module, hidden_states)

    latent_h, latent_w = split_dims(n_tokens, height, width, self_obj)
    device = hidden_states.device

    # Q, K, V
    query = module.to_q(hidden_states)
    key = module.to_k(hidden_states)
    value = module.to_v(hidden_states)

    query = module.head_to_batch_dim(query)  # (bsz*heads, N, head_dim)
    key = module.head_to_batch_dim(key)
    value = module.head_to_batch_dim(value)

    head_dim = query.shape[-1]
    scale = head_dim ** -0.5

    attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale  # (bsz*heads, N, N)

    # Prepare per-sample layouts aligned to the current hidden_states batch
    layouts = _normalize_layouts_for_batch(self_obj, bsz)

    # Build per-sample bias: (bsz, N, N)
    beta_d = float(getattr(self_obj, "beta_d", 0.5))
    bias_per_sample = torch.zeros((bsz, n_tokens, n_tokens), device=device, dtype=query.dtype)

    # NOTE: this loop is O(bsz * N^2). Usually bsz is small (1..8) and N is
    # latent tokens (e.g., 4096 at 64x64), so consider using this only on selected layers/timesteps.
    for b in range(bsz):
        layout = layouts[b]
        region_masks = _build_region_masks_from_layout(layout, latent_h, latent_w, device)
        if region_masks is None:
            # No layout -> no bias (vanilla attention for this sample)
            continue

        region_masks_f = torch.stack(region_masks, dim=0).float()  # (K, N)
        region_outer = torch.einsum("kn,km->nm", region_masks_f, region_masks_f)  # (N, N)
        allow_region = region_outer > 0

        # dropout relaxation (per sample)
        Nr = torch.rand_like(region_outer)
        allow_dropout = Nr > beta_d

        allow = allow_region | allow_dropout

        # Additive bias: 0 for allowed, -1e4 for disallowed
        bias_base = torch.zeros_like(region_outer, dtype=query.dtype, device=device)
        bias_base = bias_base.masked_fill(~allow, -1e4)

        bias_per_sample[b] = bias_base

    # Expand to (bsz*heads, N, N)
    n_heads = int(getattr(module, "heads", 1))
    bias = bias_per_sample.repeat_interleave(n_heads, dim=0)  # (bsz*heads, N, N)

    attn_scores = attn_scores + bias

    attn_probs = attn_scores.softmax(dim=-1)
    hidden_states = torch.matmul(attn_probs, value)  # (bsz*heads, N, head_dim)

    hidden_states = module.batch_to_head_dim(hidden_states)  # (bsz, N, dim)
    hidden_states = module.to_out[0](hidden_states)
    hidden_states = module.to_out[1](hidden_states)
    return hidden_states


def hook_self_forward(self_obj, module):
    """
    Wrap module.forward for self-attention (attn1).
    Applies bounded self-attention for timesteps t ∈ [t_min, t_max],
    otherwise uses vanilla self-attention.
    """
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        t_max = getattr(self_obj, "self_attn_t_max", 1000)
        t_min = getattr(self_obj, "self_attn_t_min", 200)
        t = getattr(self_obj, "cur_step", None)

        use_bounded = False
        if t is not None and (t_min <= t <= t_max):
            use_bounded = True

        if use_bounded:
            return _compute_bounded_self_attention(self_obj, module, hidden_states)
        return _vanilla_self_attention(module, hidden_states)

    return forward


def hook_forwards_self(self_obj, root_module: torch.nn.Module):
    """
    Attach self-attention (attn1) hooks to up-blocks.

    Usage in your pipeline __init__:
        from self_attention_batch import hook_forwards_self
        hook_forwards_self(self, self.unet)
    """
    for name, module in root_module.named_modules():
        if (
            "up_blocks" in name
            and "attn1" in name
            and module.__class__.__name__ == "Attention"
        ):
            module.forward = hook_self_forward(self_obj, module)
