"""Rollout recorder: one directory per episode with the exact policy inputs of every query.

    <root>/<episode_id>/episode.json         instruction, outcome
                        queries.jsonl        one line per policy query
                        control_steps.jsonl  one line per executed action
                        queries/q00000/      exterior_image.png, wrist_image.png, proprio.npz, chunk.npy
"""
import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

OUTCOMES = ("success", "failure", "intervention", "error")


class Episode:
    def __init__(self, root, instruction):
        self.id = f"ep_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        self.dir = Path(root) / self.id
        (self.dir / "queries").mkdir(parents=True)
        self.meta = {"episode_id": self.id, "task_instruction": instruction, "status": "open", "outcome": None}
        self._write_meta()
        self.queries = open(self.dir / "queries.jsonl", "a")
        self.steps = open(self.dir / "control_steps.jsonl", "a")
        self.n = 0

    def _write_meta(self):
        (self.dir / "episode.json").write_text(json.dumps(self.meta, indent=1))

    def query(self, exterior, wrist, joint_position, gripper_position, chunk, horizon):
        qid = f"{self.id}/q{self.n:05d}"
        d = self.dir / "queries" / f"q{self.n:05d}"
        d.mkdir()
        Image.fromarray(exterior).save(d / "exterior_image.png")
        Image.fromarray(wrist).save(d / "wrist_image.png")
        np.savez(d / "proprio.npz", joint_position=joint_position, gripper_position=gripper_position)
        np.save(d / "chunk.npy", np.asarray(chunk, np.float32))
        self.queries.write(json.dumps({"query_id": qid, "query_index": self.n, "intended_prefix_len": horizon}) + "\n")
        self.queries.flush()
        self.n += 1
        return qid

    def step(self, query_id, index, action):
        self.steps.write(json.dumps({"query_id": query_id, "action_index": index, "action": action.tolist()}) + "\n")
        self.steps.flush()

    def close(self, outcome):
        self.meta.update(status="closed", outcome=outcome)
        self._write_meta()
        self.queries.close()
        self.steps.close()
