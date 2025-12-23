# File: cross_attention.py
import math
import torch
import torch.nn.functional as nnF
import xformers

TOKENSCON = 77
TOKENS = 75


def _memory_efficient_attention_xformers(module, query, key, value):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
    hidden_states = module.batch_to_head_dim(hidden_states)
    return hidden_states


def main_forward_diffusers(
    module,
    hidden_states,
    encoder_hidden_states,
    divide,
    userpp=False,
    tokens=None,
    width=64,
    height=64,
    step=0,
    isxl=False,
    inhr=None,
):
    context = encoder_hidden_states
    query = module.to_q(hidden_states)
    key = module.to_k(context)
    value = module.to_v(context)

    query = module.head_to_batch_dim(query)
    key = module.head_to_batch_dim(key)
    value = module.head_to_batch_dim(value)

    hidden_states = _memory_efficient_attention_xformers(module, query, key, value)
    hidden_states = hidden_states.to(query.dtype)

    hidden_states = module.to_out[0](hidden_states)
    hidden_states = module.to_out[1](hidden_states)
    return hidden_states


def hook_forwards_cross(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "Attention":
            module.forward = hook_forward(self, module)


def _interp_hw(x_hw: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """x_hw: [B,H,W,C] -> [B,H2,W2,C]"""
    x = x_hw.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
    x = nnF.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


def _center_paste(bg: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
    """
    bg: [B,H,W,C]
    fg: [B,h,w,C] placed centered onto bg
    """
    B, H, W, C = bg.shape
    _, h, w, _ = fg.shape
    y0 = max(0, (H - h) // 2)
    x0 = max(0, (W - w) // 2)
    y1 = min(H, y0 + h)
    x1 = min(W, x0 + w)

    fg_y0 = 0
    fg_x0 = 0
    fg_y1 = y1 - y0
    fg_x1 = x1 - x0

    out = bg
    out[:, y0:y1, x0:x1, :] = fg[:, fg_y0:fg_y1, fg_x0:fg_x1, :]
    return out


def hook_forward(self, module):
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        x = hidden_states
        context = encoder_hidden_states

        height = self.h
        width = self.w

        contexts = context.clone()

        def _is_batched_split_ratio(split_ratio_obj):
            return (
                isinstance(split_ratio_obj, (list, tuple))
                and len(split_ratio_obj) > 0
                and isinstance(split_ratio_obj[0], (list, tuple))
            )

        def _is_batched_pt(pt_obj):
            if not isinstance(pt_obj, (list, tuple)) or len(pt_obj) == 0:
                return False
            first = pt_obj[0]
            if not isinstance(first, (list, tuple)) or len(first) == 0:
                return False
            if isinstance(first[0], int):
                return False
            return isinstance(first[0], (list, tuple)) and len(first[0]) == 2 and isinstance(first[0][0], int)

        def _get_per_sample(obj, b):
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                if b < len(obj):
                    return obj[b]
                return obj[-1]
            return obj

        batched_split = _is_batched_split_ratio(getattr(self, "split_ratio", None))
        batched_pt = _is_batched_pt(getattr(self, "pt", None))

        def matsepcalc_single(x1, contexts1, pn, divide, split_ratio_rows, tll):
            """
            固定模式：
              region prompt 先在 full (lh*lw) 上算 out_full，
              再等比縮放 fit region，置中貼回 region，
              剩下 padding 由 base_prompt（若啟用）填滿。
            """
            h_states = []
            x_t1 = x1.size()[1]
            (lh, lw) = split_dims(x_t1, height, width, self)

            i = 0
            outb_full = None  # [B,lh,lw,C]

            if self.usebase:
                ctx = contexts1[:, tll[i][0] * TOKENSCON : tll[i][1] * TOKENSCON, :]
                cnet_ext = contexts1.shape[1] - (contexts1.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    ctx = torch.cat([ctx, contexts1[:, -cnet_ext:, :]], dim=1)
                i += 1

                outb = main_forward_diffusers(module, x1, ctx, divide, userpp=True, isxl=self.isxl)
                outb_full = outb.reshape(outb.size()[0], lh, lw, outb.size()[2]).contiguous()

            sumout = 0
            for drow in split_ratio_rows:
                v_states = []
                sumin = 0

                # row y-range (keep your original rounding fixes)
                sumout += int(lh * drow.end) - int(lh * drow.start)
                addout = 0
                if drow.end >= 0.999:
                    addout = sumout - lh

                y0 = int(lh * drow.start) + addout
                y1 = int(lh * drow.end)
                cell_h = max(1, y1 - y0)

                for dcell in drow.cols:
                    sumin += int(lw * dcell.end) - int(lw * dcell.start)
                    addin = 0
                    if dcell.end >= 0.999:
                        addin = sumin - lw

                    x0 = int(lw * dcell.start) + addin
                    x1w = int(lw * dcell.end)
                    cell_w = max(1, x1w - x0)

                    # region ctx
                    ctx = contexts1[:, tll[i][0] * TOKENSCON : tll[i][1] * TOKENSCON, :]
                    cnet_ext = contexts1.shape[1] - (contexts1.shape[1] // TOKENSCON) * TOKENSCON
                    if cnet_ext > 0:
                        ctx = torch.cat([ctx, contexts1[:, -cnet_ext:, :]], dim=1)
                    i = i + 1 + dcell.breaks

                    # 1) full canvas out
                    out_full = main_forward_diffusers(module, x1, ctx, divide, userpp=self.pn, isxl=self.isxl)
                    out_full_hw = out_full.reshape(out_full.size()[0], lh, lw, out_full.size()[2]).contiguous()

                    # 2) compute fit scale: scale = min(cell_h/lh, cell_w/lw)
                    scale = min(float(cell_h) / float(lh), float(cell_w) / float(lw))
                    target_h = max(1, min(cell_h, int(round(lh * scale))))
                    target_w = max(1, min(cell_w, int(round(lw * scale))))

                    out_small = _interp_hw(out_full_hw, (target_h, target_w))

                    # 3) background/padding
                    if self.usebase and outb_full is not None:
                        bg = outb_full[:, y0:y1, x0:x1w, :].clone()
                        # optional base blending for the pasted content only
                        if dcell.base > 0:
                            base_small = _interp_hw(outb_full, (target_h, target_w))
                            out_small = out_small * (1 - dcell.base) + base_small * dcell.base
                    else:
                        bg = torch.zeros(
                            (out_small.size(0), cell_h, cell_w, out_small.size(3)),
                            device=out_small.device,
                            dtype=out_small.dtype,
                        )

                    # 4) center paste small into bg (padding remains bg)
                    out_cell = _center_paste(bg, out_small)
                    v_states.append(out_cell)

                output_x = torch.cat(v_states, dim=2)
                h_states.append(output_x)

            output_hw = torch.cat(h_states, dim=1)  # [B,lh,lw,C]
            output_x = output_hw.reshape(x1.size()[0], x1.size()[1], x1.size()[2])
            return output_x

        def matsepcalc_any(x2, contexts2, pn, divide):
            if (not batched_split) and (not batched_pt):
                return matsepcalc_single(x2, contexts2, pn, divide, self.split_ratio, self.pt)

            outs = []
            bsz = x2.size()[0]
            for b in range(bsz):
                split_b = _get_per_sample(self.split_ratio, b) if batched_split else self.split_ratio
                pt_b = _get_per_sample(self.pt, b) if batched_pt else self.pt
                out_b = matsepcalc_single(x2[b : b + 1], contexts2[b : b + 1], pn, divide, split_b, pt_b)
                outs.append(out_b)
            return torch.cat(outs, dim=0)

        if x.size()[0] == 1 * self.batch_size:
            output_x = matsepcalc_any(x, contexts, self.pn, 1)
        else:
            if self.isvanilla:
                nx, px = x.chunk(2)
                conn, conp = contexts.chunk(2)
            else:
                px, nx = x.chunk(2)
                conp, conn = contexts.chunk(2)

            opx = matsepcalc_any(px, conp, True, 2)
            onx = matsepcalc_any(nx, conn, False, 2)

            output_x = torch.cat([onx, opx]) if self.isvanilla else torch.cat([opx, onx])

        self.pn = not self.pn
        self.count = 0
        return output_x

    return forward


def split_dims(x_t, height, width, self=None):
    scale = math.ceil(math.log2(math.sqrt(height * width / x_t)))
    latent_h = repeat_div(height, scale)
    latent_w = repeat_div(width, scale)
    if x_t > latent_h * latent_w and hasattr(self, "nei_multi"):
        latent_h, latent_w = self.nei_multi[1], self.nei_multi[0]
        while latent_h * latent_w != x_t:
            latent_h, latent_w = latent_h // 2, latent_w // 2
    return latent_h, latent_w


def repeat_div(x, y):
    while y > 0:
        x = math.ceil(x / 2)
        y -= 1
    return x
