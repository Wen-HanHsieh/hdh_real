"""Train one task's verifier on rollouts from build_dataset.py. Omit --history for HDM.
    loss = BCEWithLogits(f(x, U), 1 - success),   Q(x, U) = 1 - sigmoid(f)

    python scripts/train.py --data data/drawer.npz --out checkpoints/drawer.pt --history
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from hdh.model import HDH, save

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--history", action="store_true")
ap.add_argument("--hmax", type=int, default=3)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

z = np.load(a.data)
tok, state, prefix, fail = z["tok"], z["state"], z["prefix"], z["fail"].astype(np.float32)
N, H, AD = prefix.shape
norm = {"action_mean": prefix.reshape(-1, AD).mean(0), "action_std": prefix.reshape(-1, AD).std(0) + 1e-6,
        "state_mean": state.mean(0), "state_std": state.std(0) + 1e-6}
pro = (state - norm["state_mean"]) / norm["state_std"]
act = (prefix - norm["action_mean"]) / norm["action_std"]
vis = tok.astype(np.float32).reshape(N, 128, -1).mean(1)

episodes = {}  # episode -> its query rows, in order
for i, e in enumerate(z["episode"]):
    episodes.setdefault(e, []).append(i)
names = sorted(episodes)
val = set(np.random.default_rng(a.seed).permutation(names)[: max(1, len(names) // 5)])
train = [episodes[e] for e in names if e not in val]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def batch(rows):
    """Context, prefixes and labels for (episode rows, position) pairs."""
    idx = [r[t] for r, t in rows]
    hist = None
    if a.history:
        n = len(rows)
        hv, hp = np.zeros((n, a.hmax, vis.shape[1]), np.float32), np.zeros((n, a.hmax, 8), np.float32)
        ha, hm = np.zeros((n, a.hmax, H, AD), np.float32), np.zeros((n, a.hmax), np.float32)
        for b, (r, t) in enumerate(rows):
            for s, j in enumerate(r[max(0, t - a.hmax):t]):
                hv[b, s], hp[b, s], ha[b, s], hm[b, s] = vis[j], pro[j], act[j], 1
        hist = tuple(torch.from_numpy(x).to(dev) for x in (hv, hp, ha, hm))
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(dev)
    return model.context(t(tok[idx]), t(pro[idx]), hist), t(act[idx]), t(fail[idx])


@torch.no_grad()
def auc():
    model.eval()
    p, y = [], []
    for e in sorted(val):
        ctx, U, label = batch([(episodes[e], t) for t in range(len(episodes[e]))])
        p.append(torch.sigmoid(model(ctx, U)).cpu().numpy())
        y.append(label.cpu().numpy())
    p, y = np.concatenate(p), np.concatenate(y)
    pos, neg = p[y == 1], p[y == 0]
    if not (len(pos) and len(neg)):
        return float("nan")
    return float((pos[:, None] > neg).mean() + 0.5 * (pos[:, None] == neg).mean())


torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
config = dict(token_dim=tok.shape[-1], horizon=H, action_dim=AD, history=a.history, hmax=a.hmax)
model = HDH(**config).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
for _ in range(a.epochs):
    model.train()
    for _ in range(max(1, sum(map(len, train)) // a.batch)):
        # episode-uniform, then query-uniform
        rows = [(train[i], rng.integers(len(train[i]))) for i in rng.integers(0, len(train), a.batch)]
        ctx, U, y = batch(rows)
        loss = F.binary_cross_entropy_with_logits(model(ctx, U), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

save(a.out, model, config, norm)
print(f"{N} queries, {len(train)} train / {len(val)} val episodes, val AUC {auc():.4f} -> {a.out}")
