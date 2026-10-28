"""Turn recorded rollouts of one task into verifier training data: for every executed query,
the policy's SigLIP prefix tokens, proprioception, executed prefix and the episode outcome.
Queries whose executed length is below H are dropped. Run in the openpi environment.

    python scripts/build_dataset.py --rollouts /data/hdh_rollouts/drawer --config pi05_droid_hdh \
        --checkpoint <ckpt> --out data/drawer.npz
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from openpi.policies import policy_config
from openpi.training import config as _config
from PIL import Image

from hdh.tokens import prefix_tokens

FAILURE = {"success": 0, "s": 0, "failure": 1, "task_failure": 1, "f": 1, "intervention": 1}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", required=True, type=Path)
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()

tokens = prefix_tokens(policy_config.create_trained_policy(_config.get_config(a.config), a.checkpoint))
data = collections.defaultdict(list)
for ep_dir in sorted(p.parent for p in a.rollouts.glob("*/episode.json")):
    ep = json.loads((ep_dir / "episode.json").read_text())
    if ep["status"] != "closed" or ep["outcome"] not in FAILURE:
        continue
    executed = collections.Counter(s["query_id"] for s in read_jsonl(ep_dir / "control_steps.jsonl"))
    for q in sorted(read_jsonl(ep_dir / "queries.jsonl"), key=lambda q: q["query_index"]):
        H = q["intended_prefix_len"]
        if executed[q["query_id"]] < H:
            continue
        d = ep_dir / "queries" / f"q{q['query_index']:05d}"
        proprio = np.load(d / "proprio.npz")
        joint, gripper = proprio["joint_position"].reshape(-1), proprio["gripper_position"].reshape(-1)
        data["tok"].append(tokens({"observation/exterior_image_1_left": np.asarray(Image.open(d / "exterior_image.png")),
                                   "observation/wrist_image_left": np.asarray(Image.open(d / "wrist_image.png")),
                                   "observation/joint_position": joint, "observation/gripper_position": gripper,
                                   "prompt": ep["task_instruction"]}).astype(np.float16))
        data["state"].append(np.concatenate([joint, gripper]).astype(np.float32))
        data["prefix"].append(np.load(d / "chunk.npy")[:H].astype(np.float32))
        data["episode"].append(ep["episode_id"])
        data["fail"].append(FAILURE[ep["outcome"]])
    print(f"{ep['episode_id']}: {ep['outcome']}", flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(a.out, **{k: np.asarray(v) for k, v in data.items()})
print(f"{len(data['fail'])} queries from {len(set(data['episode']))} episodes -> {a.out}")
