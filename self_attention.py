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

def _groups_for_cfg(layouts):
    B = len(layouts)
    if B % 2 == 0 and layouts[: B // 2] == layouts[B // 2 :]:
        return [list(range(0, B // 2)), list(range(B // 2, B))]
    return [list(range(B))]

# ---------- IoU alignment so K is consistent across frames ----------
def _iou_bool(a, b):
    inter = (a & b).sum().item(); uni = (a | b).sum().item()
    return float(inter) / float(uni) if uni else 0.0

def _align_masks_across_frames(per_frame_masks):
    """
    Align each frame's subject masks to a reference frame (the one with max regions).
    Returns: aligned[List[List[mask_k]]], K, ref_idx
    """
    B = len(per_frame_masks)
    sizes = [len(m) for m in per_frame_masks]
    b_ref = max(range(B), key=lambda i: sizes[i])
    ref = per_frame_masks[b_ref]
    K = len(ref)
    if K == 0:
        return per_frame_masks, 0, b_ref

    ref_idx = sorted(range(K), key=lambda i: ref[i].sum().item(), reverse=True)
    ref_sorted = [ref[i] for i in ref_idx]

    aligned = []
    for b in range(B):
        cur = per_frame_masks[b]
        if len(cur) == 0:
            aligned.append([torch.zeros_like(ref_sorted[0]) for _ in range(K)])
            continue
        M = torch.zeros(len(cur), K, dtype=torch.float32, device=cur[0].device)
        for i in range(len(cur)):
            for j in range(K):
                M[i, j] = _iou_bool(cur[i], ref_sorted[j])
        taken_i, taken_j, pairs = set(), set(), []
        while len(taken_j) < K and len(taken_i) < len(cur):
            idx = torch.argmax(M).item()
            i, j = idx // K, idx % K
            if (i in taken_i) or (j in taken_j):
                M.view(-1)[idx] = -1
                continue
            taken_i.add(i); taken_j.add(j); pairs.append((i, j))
            M[i, :] = -1; M[:, j] = -1
        aligned_b = [torch.zeros_like(ref_sorted[0]) for _ in range(K)]
        for i, j in pairs:
            aligned_b[j] = cur[i]
        aligned.append(aligned_b)

    aligned[b_ref] = ref_sorted
    return aligned, K, b_ref

# ---------- vanilla ----------
def _vanilla_self_attention(module, x):
    q = module.to_q(x); k = module.to_k(x); v = module.to_v(x)
    q = module.head_to_batch_dim(q); k = module.head_to_batch_dim(k); v = module.head_to_batch_dim(v)
    scale = (q.shape[-1]) ** -0.5
    scores = (q @ k.transpose(-1, -2)) * scale
    probs = scores.softmax(dim=-1)
    out = probs @ v
    out = module.batch_to_head_dim(out)
    out = module.to_out[0](out); out = module.to_out[1](out)
    return out

# ---------- Eq.(5)(6) inter-frame bounded SA ----------
def _compute_inter_bounded_self_attention(self_obj, module, hidden_states):
    """
    Implements:
      X -> reshape to (1, BN, C) -> Q,K,V
      O_hat = softmax(QK^T/sqrt(d_k) + log(M_bar)) V
      M_bar = 1[  sum_k (m^k)(m^k)^T + N_r > beta_d  ]
    with:
      - m^k ∈ {0,1}^{BN}: subject-k tokens across all frames (aligned by IoU).
      - CFG groups processed independently (no cross-group attention).
      - Optional ±K window across frames (set inter_neighbor_window=None to disable).
    """
    B, N, C = hidden_states.shape
    H_img, W_img = getattr(self_obj, "h", None), getattr(self_obj, "w", None)
    if H_img is None or W_img is None:
        return _vanilla_self_attention(module, hidden_states)

    h, w = split_dims(N, H_img, W_img, self_obj)
    device = hidden_states.device
    dtype = hidden_states.dtype

    # 1) per-frame masks
    layouts = _normalize_layouts_for_batch(self_obj, B)
    pf_masks = [(_build_region_masks_from_layout(layouts[b], h, w, device) or []) for b in range(B)]
    aligned, K, _ = _align_masks_across_frames(pf_masks)
    if K == 0:
        return _vanilla_self_attention(module, hidden_states)

    # 2) groups (CFG)
    groups = _groups_for_cfg(layouts)
    BN = B * N
    allow = torch.zeros(BN, BN, dtype=torch.bool, device=device)

    # optional frame window (not in paper; default None = all frames)
    Kwin = getattr(self_obj, "inter_neighbor_window", None)
    #print(Kwin)

    # 3) build M_bar per group (Eq.6)
    for g in groups:
        Bg = len(g)
        # indices slice map: frame index b -> BN slice [b*N : (b+1)*N]
        # build base empty (Bg*N, Bg*N)
        M_g = torch.zeros(Bg * N, Bg * N, dtype=torch.bool, device=device)

        # (a) subject-outer sum: sum_k mk mk^T
        for k_idx in range(K):
            # mk over group frames -> (Bg*N,)
            mk_parts = []
            for b in g:
                mk_b = aligned[b][k_idx] if k_idx < len(aligned[b]) else torch.zeros(N, device=device, dtype=torch.bool)
                if isinstance(Kwin, int):  # windowed: we'll still build full mk, window applied later
                    mk_parts.append(mk_b.clone())
                else:
                    mk_parts.append(mk_b)
            mk = torch.cat(mk_parts, dim=0)  # (Bg*N,)
            if not mk.any():
                continue
            Mk = torch.outer(mk, mk)  # (Bg*N, Bg*N) bool

            # if isinstance(Kwin, int):
            #     # apply ±K window on frame blocks
            #     mask_win = torch.zeros_like(Mk)
            #     for i in range(Bg):
            #         for j in range(max(0, i - Kwin), min(Bg, i + Kwin + 1)):
            #             si, sj = slice(i * N, (i + 1) * N), slice(j * N, (j + 1) * N)
            #             mask_win[si, sj] = True
            #     Mk &= mask_win
            M_g |= Mk

        # (b) dropout-attention relaxation: Nr > beta_d
        beta_d = float(getattr(self_obj, "beta_d", 0.5))
        Nr = torch.rand((Bg * N, Bg * N), device=device)
        M_g |= (Nr > beta_d)

        # (c) ensure diagonal True (numerical stability)
        M_g.fill_diagonal_(True)

        # write M_g back to global allow
        for i, bi in enumerate(g):
            for j, bj in enumerate(g):
                gi = slice(bi * N, (bi + 1) * N)
                gj = slice(bj * N, (bj + 1) * N)
                li = slice(i * N, (i + 1) * N)
                lj = slice(j * N, (j + 1) * N)
                allow[gi, gj] = M_g[li, lj]

    # 4) joint attention over (BN) tokens (Eq.5)
    x = hidden_states.reshape(1, BN, C)

    q = module.to_q(x); k = module.to_k(x); v = module.to_v(x)
    q = module.head_to_batch_dim(q); k = module.head_to_batch_dim(k); v = module.head_to_batch_dim(v)

    scale = (q.shape[-1]) ** -0.5
    scores = (q @ k.transpose(-1, -2)) * scale

    # log(M_bar) ≈ add large negative bias where not allowed
    bias = torch.zeros(BN, BN, dtype=q.dtype, device=device).masked_fill(~allow, -1e4)
    n_heads = int(getattr(module, "heads", 1))
    scores = scores + bias.expand(n_heads, BN, BN)

    probs = scores.softmax(dim=-1)
    out = probs @ v
    out = module.batch_to_head_dim(out).reshape(B, N, -1)

    out = module.to_out[0](out)
    out = module.to_out[1](out)
    return out

# ---------- hooks ----------
def hook_self_forward(self_obj, module):
    """
    Replace Attention.forward (self-attn/attn1). Active only for t in [t_min, t_max].
    """
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        t_max = getattr(self_obj, "self_attn_t_max", 1000)
        t_min = getattr(self_obj, "self_attn_t_min", 200)
        t = getattr(self_obj, "cur_step", None)
        if (t is not None) and (t_min <= t <= t_max):
            return _compute_inter_bounded_self_attention(self_obj, module, hidden_states)
        return _vanilla_self_attention(module, hidden_states)
    return forward

def hook_forwards_self(self_obj, root_module: torch.nn.Module):
    """
    Attach to self-attn ('attn1'). You can expand to down/mid as needed.
    """
    for name, module in root_module.named_modules():
        if ("attn1" in name) and (module.__class__.__name__ == "Attention") and (("up_blocks" in name) or ("mid_block" in name) or ("down_blocks" in name)):
            module.forward = hook_self_forward(self_obj, module)
