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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
import re

import math
import torch
import xformers  # noqa: F401  (kept for parity; we use torch matmul here)
from cross_attention import split_dims


def _normalize_subject_ids_for_batch(self_obj, bsz: int):
    """
    Return subject_ids with length == bsz.
    subject_ids[b] is a list[str] for that sample's BREAK segments (or region count later).
    """
    ids = getattr(self_obj, "inter_subject_ids_per_sample", None)
    if ids is None:
        return None

    if not isinstance(ids, (list, tuple)) or len(ids) == 0:
        return None

    n = len(ids)
    if n == bsz:
        return list(ids)
    if bsz == 2 * n:
        return list(ids) + list(ids)  # CFG duplicate
    # fallback repeat
    return [ids[i % n] for i in range(bsz)]
def _debug_print_region_subject_alignment_from_ids(
    aligned_masks,
    subject_ids,   # <-- 直接傳進來
):
    print("\n[Inter-SA][DEBUG] ===== Region ↔ Subject Alignment =====")
    for b, masks in enumerate(aligned_masks):
        print(f"[Frame {b}]")
        ids_b = subject_ids[b] if b < len(subject_ids) else []
        for k, m in enumerate(masks):
            area = int(m.sum().item())
            sid = ids_b[k] if k < len(ids_b) else "bg"   # <-- 這裡也別印 MISSING，補 bg
            print(f"  region k={k:<2d}  subject={sid:<5s}  area={area}")
    print("[Inter-SA][DEBUG] =====================================\n")


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
    #print("len(layoyt):", B)
    if B % 2 == 0 and layouts[: B // 2] == layouts[B // 2 :]:
        return [list(range(0, B // 2)), list(range(B // 2, B))]
    return [list(range(B))]

# ---------- IoU alignment so K is consistent across frames ----------
def _iou_bool(a, b):
    inter = (a & b).sum().item(); uni = (a | b).sum().item()
    return float(inter) / float(uni) if uni else 0.0


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

def _normalize_subject_ids_per_sample(self_obj, B: int, region_masks_per_frame: List[List[torch.Tensor]]):
    """
    Returns subject_ids_per_frame aligned to number of masks per frame:
      subject_ids[b] length == len(region_masks_per_frame[b])
    Source priority:
      1) self_obj.inter_subject_ids_per_sample (from pipeline)
      2) fallback: assume first inter_subject_k are subjects, rest bg
    """
    inter_k = int(getattr(self_obj, "inter_subject_k", 2))
    subject_ids_in = getattr(self_obj, "inter_subject_ids_per_sample", None)

    out: List[List[str]] = []
    for b in range(B):
        mlen = len(region_masks_per_frame[b])
        if isinstance(subject_ids_in, (list, tuple)) and len(subject_ids_in) == B and isinstance(subject_ids_in[b], (list, tuple)):
            ids_b = list(subject_ids_in[b])
        else:
            # fallback: first inter_k are "sub0/sub1", rest bg
            ids_b = [f"sub{i}" for i in range(min(inter_k, mlen))] + ["bg"] * max(0, mlen - inter_k)

        # trim/pad to masks length
        if len(ids_b) < mlen:
            ids_b = ids_b + ["bg"] * (mlen - len(ids_b))
        if len(ids_b) > mlen:
            ids_b = ids_b[:mlen]
        out.append(ids_b)
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


    # 先做 CFG duplication（5->10）
    subject_ids_per_frame_in = _normalize_subject_ids_for_batch(self_obj, B)

    # 把 region_masks 建好（每個 frame 的實際 region 數）
    region_masks = [(_build_region_masks_from_layout(layouts[b], h, w, device) or []) for b in range(B)]

    # 用 duplicated 的 subject id 來 pad/trim 到每個 frame 的 region 數
    _old = getattr(self_obj, "inter_subject_ids_per_sample", None)
    if subject_ids_per_frame_in is not None:
        self_obj.inter_subject_ids_per_sample = subject_ids_per_frame_in

    subject_ids = _normalize_subject_ids_per_sample(self_obj, B, region_masks)

    self_obj.inter_subject_ids_per_sample = _old

    

    # 2) groups (CFG)
    groups = _groups_for_cfg(layouts)
    BN = B * N
    allow = torch.zeros(BN, BN, dtype=torch.bool, device=device)

    # optional frame window (not in paper; default None = all frames)
    Kwin = getattr(self_obj, "inter_neighbor_window", None)
    allow = torch.zeros((BN, BN), dtype=torch.bool, device=device)
    #print(Kwin)

    # 3) build M_bar per group (Eq.6)
    for g in groups:
        Bg = len(g) # of frame = 5
        # indices slice map: frame index b -> BN slice [b*N : (b+1)*N]
        # build base empty (Bg*N, Bg*N)
        M_g = torch.zeros(Bg * N, Bg * N, dtype=torch.bool, device=device)
        subj_set = []
        #print("subject_ids", subject_ids)
        for bi in g:
            for sid in subject_ids[bi]:
                if sid == "bg":#background
                    continue
                if sid not in subj_set:
                    subj_set.append(sid)
        K_subject = int(getattr(self_obj, "inter_subject_k", 2))
        #print("subj_set:", subj_set)
        # (a) subject-outer sum: sum_k mk mk^T
        for k_idx in subj_set:
            # mk over group frames -> (Bg*N,)
            mk_parts = []
            for b in g:
                # union all region masks in this frame that match subject k_idx
                mk_b = torch.zeros((N,), dtype=torch.bool, device=device)
                for ridx, mk in enumerate(region_masks[b]):
                    if subject_ids[b][ridx] == k_idx and mk is not None:
                        mk_b |= mk
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
