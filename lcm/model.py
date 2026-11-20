"""The Large Complexity Model (Phase 0 size).

Tokens: one per (entity, time patch). A patch is PATCH consecutive time steps
of all of that entity's features - the "word" of a complex system.

Attention is axial: each block first looks back along an entity's own history
(causal in time, like an LLM), then across all entities at the same moment,
with learned per-head bonuses for "self" and "directly interacting entity",
so the interaction graph guides attention without being required. Stacked
blocks let information flow from any entity's past to any other.

Output: for every token, a mixture of Gaussians over the entity's *next* patch.
Sampling patch after patch gives whole possible futures, and counting how
often an outcome happens across samples gives its likelihood.

The model is never told which field a system comes from. It has to work it out
from the history it is shown - the in-context learning that makes LLMs general.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NMAX, FMAX, PATCH, T_TOTAL

TP_MAX = T_TOTAL // PATCH
PF = PATCH * FMAX


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, mask=None, causal=False):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if mask is not None:
            mask = mask.to(q.dtype)          # matches q under bf16 autocast; no-op in fp32
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
        return x + self.proj(a.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    """Axial block: look back in time (per entity), then across entities (per moment)."""

    def __init__(self, d, heads):
        super().__init__()
        self.time = Attn(d, heads)
        self.ent = Attn(d, heads)
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h, ent_bias):
        B, N, Tp, D = h.shape
        h = self.time(h.reshape(B * N, Tp, D), causal=True).view(B, N, Tp, D)
        h = h.transpose(1, 2).reshape(B * Tp, N, D)
        h = self.ent(h, mask=ent_bias).view(B, Tp, N, D).transpose(1, 2)
        return h + self.mlp(self.ln(h))


class LCM(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, K=4):
        super().__init__()
        self.K, self.heads = K, heads
        self.embed = nn.Linear(PF + FMAX + 2 * FMAX, d)
        self.pos = nn.Embedding(TP_MAX, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.b_self = nn.Parameter(torch.zeros(layers, heads))
        self.b_adj = nn.Parameter(torch.zeros(layers, heads))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, K + 2 * K * PF)

    @staticmethod
    def to_patches(xn):
        B, T, N, Fd = xn.shape
        return xn.view(B, T // PATCH, PATCH, N, Fd).permute(0, 3, 1, 2, 4).reshape(B, N, T // PATCH, PATCH * Fd)

    @staticmethod
    def from_patches(p):
        B, N, Tp, _ = p.shape
        return p.view(B, N, Tp, PATCH, FMAX).permute(0, 2, 3, 1, 4).reshape(B, Tp * PATCH, N, FMAX)

    def forward(self, xn, ent_mask, feat_mask, adj, stats):
        """xn [B, T, N, F] normalised history -> mixture over each token's next patch."""
        B = xn.shape[0]
        patches = self.to_patches(xn)                                  # [B, N, Tp, PF]
        Tp = patches.shape[2]
        fm_pf = feat_mask.repeat(1, PATCH)                             # [B, PF]
        side = torch.cat([feat_mask, stats], -1)[:, None, None].expand(B, NMAX, Tp, -1)
        h = self.embed(torch.cat([patches * fm_pf[:, None, None], side], -1))
        h = h + self.pos.weight[:Tp][None, None]                     # [B, N, Tp, d]

        # entity-attention bias: hide missing entities, learned bonus for self / neighbours
        eye = torch.eye(NMAX, device=xn.device)
        hidden = (ent_mask[:, None, :] == 0) & (eye[None] == 0)
        base = torch.zeros(B, 1, NMAX, NMAX, device=xn.device).masked_fill(hidden[:, None], float("-inf"))
        for li, blk in enumerate(self.blocks):
            bias = (base + self.b_self[li][None, :, None, None] * eye
                    + self.b_adj[li][None, :, None, None] * adj[:, None])
            bias = bias.repeat_interleave(Tp, 0)                       # one copy per time patch
            h = blk(h, bias)

        # the forecast head stays in full precision even when the transformer
        # runs in bf16, so means and widths of the bell curves aren't rounded
        with torch.autocast(device_type=h.device.type, enabled=False):
            out = self.head(self.ln(h.float()))
        logits = out[..., : self.K].log_softmax(-1)
        mu, logsig = out[..., self.K:].view(B, NMAX, Tp, 2, self.K, PF).unbind(3)
        return logits, mu, logsig.clamp(-7, 3)

    def loss(self, xn, ent_mask, feat_mask, adj, stats):
        """Average negative log-likelihood of every next patch (like LLM next-token loss)."""
        logits, mu, logsig = self(xn, ent_mask, feat_mask, adj, stats)
        target = self.to_patches(xn)[:, :, 1:]                         # [B, N, Tp-1, PF]
        logits, mu, logsig = logits[:, :, :-1], mu[:, :, :-1], logsig[:, :, :-1]
        m = feat_mask.repeat(1, PATCH)[:, None, None, None, :]         # valid values
        z = (target[..., None, :] - mu) * torch.exp(-logsig)
        lp = (-0.5 * z ** 2 - logsig - 0.5 * math.log(2 * math.pi)) * m
        ll = torch.logsumexp(logits + lp.sum(-1), -1)                  # [B, N, Tp-1]
        w = ent_mask[:, :, None].expand_as(ll)
        n_vals = (w * feat_mask.sum(-1)[:, None, None] * PATCH).sum()
        return -(ll * w).sum() / n_vals

    @torch.no_grad()
    def sample_future(self, ctx, ent_mask, feat_mask, adj, stats, horizon, n_samples, temp=1.0):
        """Draw n_samples possible futures. ctx [B, Tc, N, F] -> [B, S, horizon, N, F]."""
        B = ctx.shape[0]
        rep = lambda z: z.repeat_interleave(n_samples, 0)
        x, em, fm, a, st = rep(ctx), rep(ent_mask), rep(feat_mask), rep(adj), rep(stats)
        fm_pf = fm.repeat(1, PATCH)[:, None]
        for _ in range(horizon // PATCH):
            logits, mu, logsig = self(x, em, fm, a, st)
            logits, mu, logsig = logits[:, :, -1], mu[:, :, -1], logsig[:, :, -1]   # next patch
            k = torch.distributions.Categorical(logits=logits).sample()             # [B*S, N]
            pick = lambda z: z.gather(2, k[..., None, None].expand(-1, -1, 1, PF)).squeeze(2)
            nxt = pick(mu) + temp * torch.exp(pick(logsig)) * torch.randn_like(pick(mu))
            nxt = (nxt * fm_pf * em[..., None]).clamp(-10, 10)
            x = torch.cat([x, self.from_patches(nxt[:, :, None])], 1)
        fut = x[:, ctx.shape[1]:]
        return fut.view(B, n_samples, *fut.shape[1:])
