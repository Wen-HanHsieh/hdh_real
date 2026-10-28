import os

import numpy as np
import torch
import torch.nn as nn


class HDH(nn.Module):
    """Action verifier. Returns the failure logit f(x, U); Q(x, U) = 1 - sigmoid(f).

    x: SigLIP prefix tokens of two camera views (2 x 8 x 8 x token_dim), proprioception,
       and, with history=True, the last `hmax` committed queries (HDH). Without history
       the model is HDM.
    U: the executed prefix of a candidate chunk (horizon x action_dim).
    """

    def __init__(self, token_dim=2048, horizon=8, action_dim=7, history=True, hmax=3, d=128):
        super().__init__()
        self.history, self.hmax = history, hmax
        self.vproj = nn.Linear(token_dim, d)
        self.spat = nn.Parameter(torch.zeros(64, d))
        self.view = nn.Parameter(torch.zeros(2, d))
        self.pproj = nn.Linear(8, d)
        self.task = nn.Parameter(torch.zeros(d))
        self.aproj = nn.Linear(action_dim, d)
        self.apos = nn.Parameter(torch.zeros(horizon, d))
        if history:
            self.h_vis = nn.Linear(token_dim, d)
            self.h_pro = nn.Linear(8, d)
            self.h_act = nn.Linear(action_dim, d)
            self.gru = nn.GRU(d, d, batch_first=True)
            self.h_none = nn.Parameter(torch.zeros(d))
        self.sattn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.lns = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def context(self, tok, pro, hist=None):
        """Observation (+ history) tokens that candidate actions attend to."""
        B = tok.shape[0]
        v = self.vproj(tok.reshape(B, 2, 64, -1)) + self.spat + self.view[:, None]
        ctx = [v.reshape(B, 128, -1), self.pproj(pro)[:, None], self.task.expand(B, 1, -1)]
        if self.history and hist is not None:
            ctx.append(self.history_token(*hist)[:, None])
        return torch.cat(ctx, 1)

    def history_token(self, vis, pro, act, mask):
        """GRU over up to hmax past queries: pooled tokens, proprio, executed actions."""
        e = (self.h_vis(vis) + self.h_pro(pro) + self.h_act(act.mean(2))) * mask[..., None]
        out, _ = self.gru(e)
        n = mask.sum(1).long()
        h = out[torch.arange(len(out), device=out.device), (n - 1).clamp(min=0)]
        return torch.where((n == 0)[:, None], self.h_none.expand_as(h), h)

    def forward(self, ctx, actions):
        q = self.aproj(actions) + self.apos
        q = self.lns(q + self.sattn(q, q, q)[0])
        h = self.ln1(q + self.attn(q, ctx, ctx)[0])
        h = self.ln2(h + self.ff(h))
        return self.head(h.mean(1)).squeeze(-1)


def save(path, model, config, norm):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"config": config, "state_dict": model.state_dict(),
                "norm": {k: torch.as_tensor(v, dtype=torch.float32) for k, v in norm.items()}}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    model = HDH(**ck["config"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, {k: v.cpu().numpy() for k, v in ck["norm"].items()}


def pool(tok):
    """Mean-pool one query's prefix tokens into the history visual feature."""
    return np.asarray(tok, np.float32).reshape(128, -1).mean(0)
