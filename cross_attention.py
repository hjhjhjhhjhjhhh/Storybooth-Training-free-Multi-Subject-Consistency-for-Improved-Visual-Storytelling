import math
from pprint import pprint
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, Resize 
import xformers
TOKENSCON = 77
TOKENS = 75



def _memory_efficient_attention_xformers(module, query, key, value):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    hidden_states = xformers.ops.memory_efficient_attention(query, key, value,attn_bias=None)
    hidden_states = module.batch_to_head_dim(hidden_states)
    return hidden_states

def main_forward_diffusers(module,hidden_states,encoder_hidden_states,divide,userpp = False,tokens=[],width = 64,height = 64,step = 0, isxl = False, inhr = None):
    context = encoder_hidden_states
    query = module.to_q(hidden_states)
    # cond, uncond =query.chunk(2)
    # query=torch.cat([cond,uncond])
    key = module.to_k(context)
    value = module.to_v(context)
    query = module.head_to_batch_dim(query)
    key = module.head_to_batch_dim(key)
    value = module.head_to_batch_dim(value)
    hidden_states=_memory_efficient_attention_xformers(module, query, key, value)
    hidden_states = hidden_states.to(query.dtype)
    # linear proj
    hidden_states = module.to_out[0](hidden_states)
    # dropout
    hidden_states = module.to_out[1](hidden_states)
    return hidden_states
    
def hook_forwards_cross(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "Attention":
            # print(f"Attaching hook to {name}")
            module.forward = hook_forward(self, module)           


def hook_forward(self, module):
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        x = hidden_states
        context = encoder_hidden_states

        height = self.h
        width = self.w

        # Keep the original latent dim inference logic intact.
        x_t = x.size()[1]
        scale = round(math.sqrt(height * width / x_t))
        latent_h = round(height / scale)
        latent_w = round(width / scale)
        ha, wa = x_t % latent_h, x_t % latent_w
        if ha == 0:
            latent_w = int(x_t / latent_h)
        elif wa == 0:
            latent_h = int(x_t / latent_w)

        contexts = context.clone()

        # -------------------------
        # Batch helpers
        # -------------------------
        def _is_batched_split_ratio(split_ratio_obj):
            # Single: [Row, Row, ...]
            # Batched: [[Row, ...], [Row, ...], ...]
            return isinstance(split_ratio_obj, (list, tuple)) and len(split_ratio_obj) > 0 and isinstance(split_ratio_obj[0], (list, tuple))

        def _is_batched_pt(pt_obj):
            # Single: [(s,e), (s,e), ...]
            # Batched: [[(s,e), ...], [(s,e), ...], ...]
            if not isinstance(pt_obj, (list, tuple)) or len(pt_obj) == 0:
                return False
            first = pt_obj[0]
            if not isinstance(first, (list, tuple)) or len(first) == 0:
                return False
            # If first element is itself a span (int,int), this is single.
            if isinstance(first[0], int):
                return False
            # If first element contains spans, it's batched.
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
            """Original matsepcalc logic, but parameterized by split_ratio_rows / tll.
            x1 and contexts1 may be batch-sized (single mode) or batch=1 (batched mode).
            """
            h_states = []
            x_t1 = x1.size()[1]
            (lh, lw) = split_dims(x_t1, height, width, self)

            latent_out = lw
            latent_in = lh

            i = 0
            outb = None

            if self.usebase:
                ctx = contexts1[:, tll[i][0] * TOKENSCON : tll[i][1] * TOKENSCON, :]
                cnet_ext = contexts1.shape[1] - (contexts1.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    ctx = torch.cat([ctx, contexts1[:, -cnet_ext:, :]], dim=1)

                i = i + 1
                out = main_forward_diffusers(module, x1, ctx, divide, userpp=True, isxl=self.isxl)

                outb = out.clone()
                outb = outb.reshape(outb.size()[0], lh, lw, outb.size()[2])

            sumout = 0

            for drow in split_ratio_rows:
                v_states = []
                sumin = 0
                for dcell in drow.cols:
                    # Grab token block for this region.
                    ctx = contexts1[:, tll[i][0] * TOKENSCON : tll[i][1] * TOKENSCON, :]

                    # ControlNet sends extra conds at the end of context, apply it to all regions.
                    cnet_ext = contexts1.shape[1] - (contexts1.shape[1] // TOKENSCON) * TOKENSCON
                    if cnet_ext > 0:
                        ctx = torch.cat([ctx, contexts1[:, -cnet_ext:, :]], dim=1)

                    i = i + 1 + dcell.breaks

                    out = main_forward_diffusers(module, x1, ctx, divide, userpp=self.pn, isxl=self.isxl)

                    out = out.reshape(out.size()[0], lh, lw, out.size()[2])  # to [B, H, W, C]

                    addout = 0
                    addin = 0
                    sumin = sumin + int(latent_in * dcell.end) - int(latent_in * dcell.start)
                    if dcell.end >= 0.999:
                        addin = sumin - latent_in
                        sumout = sumout + int(latent_out * drow.end) - int(latent_out * drow.start)
                        if drow.end >= 0.999:
                            addout = sumout - latent_out

                    out = out[
                        :,
                        int(lh * drow.start) + addout : int(lh * drow.end),
                        int(lw * dcell.start) + addin : int(lw * dcell.end),
                        :,
                    ]

                    if self.usebase:
                        outb_t = outb[
                            :,
                            int(lh * drow.start) + addout : int(lh * drow.end),
                            int(lw * dcell.start) + addin : int(lw * dcell.end),
                            :,
                        ].clone()
                        out = out * (1 - dcell.base) + outb_t * dcell.base

                    v_states.append(out)

                output_x = torch.cat(v_states, dim=2)  # concat cells -> row
                h_states.append(output_x)

            output_x = torch.cat(h_states, dim=1)  # concat rows -> full layer
            output_x = output_x.reshape(x1.size()[0], x1.size()[1], x1.size()[2])  # back to [B, T, C]
            return output_x

        def matsepcalc_any(x2, contexts2, pn, divide):
            """Run either in single-batch mode (original) or per-sample mode (new batched split_ratio/pt)."""
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

        # -------------------------
        # Guidance handling (original structure)
        # -------------------------
        if x.size()[0] == 1 * self.batch_size:
            output_x = matsepcalc_any(x, contexts, self.pn, 1)
        else:
            if self.isvanilla:  # SBM DDIM reverses cond/uncond.
                nx, px = x.chunk(2)
                conn, conp = contexts.chunk(2)
            else:
                px, nx = x.chunk(2)
                conp, conn = contexts.chunk(2)

            opx = matsepcalc_any(px, conp, True, 2)
            onx = matsepcalc_any(nx, conn, False, 2)

            if self.isvanilla:
                output_x = torch.cat([onx, opx])
            else:
                output_x = torch.cat([opx, onx])

        self.pn = not self.pn
        self.count = 0
        return output_x

    return forward


def split_dims(x_t, height, width, self=None):
    """Split an attention layer dimension to height + width.
    The original estimate was latent_h = sqrt(hw_ratio*x_t),
    rounding to the nearest value. However, this proved inaccurate.
    The actual operation seems to be as follows:
    - Divide h,w by 8, rounding DOWN.
    - For every new layer (of 4), divide both by 2 and round UP (then back up).
    - Multiply h*w to yield x_t.
    There is no inverse function to this set of operations,
    so instead we mimic them without the multiplication part using the original h+w.
    It's worth noting that no known checkpoints follow a different system of layering,
    but it's theoretically possible. Please report if encountered.
    """
    scale = math.ceil(math.log2(math.sqrt(height * width / x_t)))
    latent_h = repeat_div(height, scale)
    latent_w = repeat_div(width, scale)
    if x_t > latent_h * latent_w and hasattr(self, "nei_multi"):
        latent_h, latent_w = self.nei_multi[1], self.nei_multi[0] 
        while latent_h * latent_w != x_t:
            latent_h, latent_w = latent_h // 2, latent_w // 2

    return latent_h, latent_w

def repeat_div(x,y):
    """Imitates dimension halving common in convolution operations.
    
    This is a pretty big assumption of the model,
    but then if some model doesn't work like that it will be easy to spot.
    """
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x