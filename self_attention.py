# self_attention.py

import math
import torch
import xformers
from cross_attention import split_dims

def _build_region_masks_from_split_ratio(self_obj, latent_h, latent_w, device):
    """
    Build per-region token masks from self.split_ratio (RPG layout).
    Each region mask is a flattened bool tensor of length N = latent_h * latent_w.
    We assume self.split_ratio is the same structure used in cross_attention.py:
      - A list of rows, each 'drow' has .start, .end, .cols
      - Each 'dcell' in drow.cols has .start, .end, etc. :contentReference[oaicite:4]{index=4}
    """
    if not hasattr(self_obj, "split_ratio") or self_obj.split_ratio is None:
        return None

    region_masks = []
    for drow in self_obj.split_ratio:
        row_start = int(latent_h * drow.start)
        row_end = int(latent_h * drow.end)

        for dcell in drow.cols:
            col_start = int(latent_w * dcell.start)
            col_end = int(latent_w * dcell.end)

            # safeguard against degenerate / rounding issues
            row_start_clamped = max(0, min(latent_h, row_start))
            row_end_clamped = max(0, min(latent_h, row_end))
            col_start_clamped = max(0, min(latent_w, col_start))
            col_end_clamped = max(0, min(latent_w, col_end))

            if row_end_clamped <= row_start_clamped or col_end_clamped <= col_start_clamped:
                continue

            mask = torch.zeros(latent_h, latent_w, dtype=torch.bool, device=device)
            mask[row_start_clamped:row_end_clamped, col_start_clamped:col_end_clamped] = True
            region_masks.append(mask.view(-1))  # (N,)

    if len(region_masks) == 0:
        return None

    return region_masks


def _compute_bounded_self_attention(
    self_obj,
    module,
    hidden_states,
):
    """
    Implements bounded self-attention with dropout as in StoryBooth:
      A_l = softmax(Q_l K_l^T / sqrt(d_k) + log(M_l))
      O_l = A_l V_l,
    where M_l encodes region-constrained attention + dropout. :contentReference[oaicite:5]{index=5}
    """
    bsz, n_tokens, _ = hidden_states.shape
    height = getattr(self_obj, "h", None)
    width = getattr(self_obj, "w", None)

    # Fallback: if we don't know spatial size, just do vanilla attention
    if height is None or width is None:
        return _vanilla_self_attention(module, hidden_states)

    latent_h, latent_w = split_dims(n_tokens, height, width, self_obj)
    device = hidden_states.device

    # 1. Compute Q, K, V as in diffusers Attention
    query = module.to_q(hidden_states)
    key = module.to_k(hidden_states)
    value = module.to_v(hidden_states)

    query = module.head_to_batch_dim(query)  # (bsz * heads, N, head_dim)
    key = module.head_to_batch_dim(key)
    value = module.head_to_batch_dim(value)

    head_dim = query.shape[-1]
    scale = head_dim ** -0.5

    # 2. Base attention scores
    attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale  # (bsz*heads, N, N)

    # 3. Build region masks (per image), then construct Ml as in Eq. 4 with dropout
    region_masks = _build_region_masks_from_split_ratio(self_obj, latent_h, latent_w, device)
    # If no layout info, fall back to vanilla self-attention
    if region_masks is None:
        attn_probs = attn_scores.softmax(dim=-1)
        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = module.batch_to_head_dim(hidden_states)
        hidden_states = module.to_out[0](hidden_states)
        hidden_states = module.to_out[1](hidden_states)
        return hidden_states

    # Stack region masks: (K, N)
    region_masks = torch.stack(region_masks, dim=0).float()
    # Σ_k (m̄_k m̄_k^T) -> (N, N) matrix counting how many regions share token-pairs
    # We only care about >0 vs ==0. :contentReference[oaicite:6]{index=6}
    region_outer = torch.einsum("kn,km->nm", region_masks, region_masks)
    allow_region = region_outer > 0  # True if tokens belong to same subject region

    # dropout matrix Nr and threshold β_d
    beta_d = getattr(self_obj, "beta_d", 0.5)
    Nr = torch.rand_like(region_outer)
    allow_dropout = Nr > beta_d

    allow = allow_region | allow_dropout  # Eq. (4)

    # convert allow mask -> additive bias (0 for allowed, large negative for disallowed)
    # we don't literally do log(M); we emulate it with a big negative constant.
    bias_base = torch.zeros_like(region_outer, dtype=query.dtype)
    bias_base = bias_base.masked_fill(~allow, -1e4)

    # Now we need a bias per (batch, head). The layout is shared across samples, so copy.
    n_heads = module.heads
    # attn_scores.shape[0] == bsz * heads
    bias_per_sample = bias_base.unsqueeze(0).expand(bsz, -1, -1)  # (bsz, N, N)
    bias = bias_per_sample.repeat_interleave(n_heads, dim=0)      # (bsz*heads, N, N)

    attn_scores = attn_scores + bias

    # 4. Softmax + value projection
    attn_probs = attn_scores.softmax(dim=-1)
    hidden_states = torch.matmul(attn_probs, value)  # (bsz*heads, N, head_dim)

    # 5. Merge heads and final linear + dropout
    hidden_states = module.batch_to_head_dim(hidden_states)  # (bsz, N, dim)
    hidden_states = module.to_out[0](hidden_states)
    hidden_states = module.to_out[1](hidden_states)
    return hidden_states


def _vanilla_self_attention(module, hidden_states):
    """
    Plain self-attention with the same math as diffusers' Attention,
    but without any bounding. Used outside the specified timestep range
    or when we lack layout info.
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


def hook_self_forward(self_obj, module):
    """
    Wraps module.forward for self-attention (attn1) layers.
    Applies bounded self-attention predominantly on up-blocks and
    only for timesteps t ∈ [1000, 200] (reverse diffusion range),
    otherwise falls back to vanilla self-attention.
    """

    # You can configure these on the pipeline object if you want different ranges.
    def forward(
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        t_max = getattr(self_obj, "self_attn_t_max", 1000)
        t_min = getattr(self_obj, "self_attn_t_min", 200)
        t = self_obj.cur_step

        # If timestep in [t_min, t_max], apply bounded self-attention; else vanilla.
        use_bounded = False
        if t is not None:
            if t_min <= t <= t_max:
                use_bounded = True

        if use_bounded:
            return _compute_bounded_self_attention(self_obj, module, hidden_states)
        else:
            return _vanilla_self_attention(module, hidden_states)

    return forward


def hook_forwards_self(self_obj, root_module: torch.nn.Module):
    """
    Attach self-attention (attn1) hooks to up-blocks.

    Usage in your RegionalDiffusionXLPipeline __init__:
        from self_attention import hook_forwards_self
        hook_forwards_self(self, self.unet)
    """
    for name, module in root_module.named_modules():
        # Only modify self-attention in up-blocks, and only Attention modules.
        if (
            "up_blocks" in name
            and "attn1" in name
            and module.__class__.__name__ == "Attention"
        ):
            module.forward = hook_self_forward(self_obj, module)
