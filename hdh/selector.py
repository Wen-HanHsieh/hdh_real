import os

import numpy as np
import torch

from hdh.model import load, pool

TASKS = ("plate", "drawer", "closet")


class Selector:
    """Per-task HDH verifiers, routed by instruction keyword. Inputs are built exactly as in
    scripts/train.py: f16 tokens, normalized state and actions, and a history of the last
    `hmax` committed queries (empty at the first query of an episode)."""

    def __init__(self, ckpt_dir, tokens, device="cuda"):
        self.tokens, self.device = tokens, device
        self.nets = {t: load(f"{ckpt_dir}/{t}.pt", device) for t in TASKS if os.path.exists(f"{ckpt_dir}/{t}.pt")}
        assert self.nets, f"no verifier ({', '.join(f'{t}.pt' for t in TASKS)}) in {ckpt_dir}"
        self.hist = []

    def _tensor(self, x):
        return torch.as_tensor(np.asarray(x, np.float32)[None], device=self.device)

    def _history(self, model, width):
        n, horizon, action_dim = model.hmax, model.apos.shape[0], model.aproj.in_features
        vis, pro = np.zeros((n, width), np.float32), np.zeros((n, 8), np.float32)
        act, mask = np.zeros((n, horizon, action_dim), np.float32), np.zeros(n, np.float32)
        for i, (v, p, a) in enumerate(self.hist[-n:]):
            vis[i], pro[i], act[i], mask[i] = v, p, a, 1.0
        return tuple(self._tensor(x) for x in (vis, pro, act, mask))

    @torch.no_grad()
    def select(self, obs, candidates, horizon, query_index):
        """Index of the candidate to execute and its record, or None if no verifier matches the prompt."""
        if query_index == 0:
            self.hist = []
        task = next((t for t in self.nets if t in obs["prompt"].lower()), None)
        if task is None:
            return None
        model, nm = self.nets[task]
        tok = self.tokens(obs).astype(np.float16)
        state = np.concatenate([obs["observation/joint_position"], obs["observation/gripper_position"]], axis=None)
        pro = ((state - nm["state_mean"]) / nm["state_std"]).astype(np.float32)
        U = np.stack([np.asarray(c, np.float32)[:horizon] for c in candidates])
        U = ((U - nm["action_mean"]) / nm["action_std"]).astype(np.float32)
        hist = self._history(model, tok.shape[-1]) if model.history else None
        ctx = model.context(self._tensor(tok), self._tensor(pro), hist)
        p_fail = torch.sigmoid(model(ctx.repeat(len(U), 1, 1), torch.as_tensor(U, device=self.device))).cpu().numpy()
        k = int(np.argmin(p_fail))
        self.hist = (self.hist + [(pool(tok), pro, U[k])])[-model.hmax:]
        return k, {"task": task, "selected": k, "q": (1 - p_fail).tolist()}
