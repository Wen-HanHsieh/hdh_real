"""DROID rollout client (runs on the robot workstation). Based on openpi examples/droid/main.py.

    python robot/main.py --host <server> --noise_seed 1000 --record_dir /data/hdh_rollouts
"""
import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tyro
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from recorder import OUTCOMES, Episode

HZ = 15


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    external_camera: str = "34520144"  # must match the camera used for fine-tuning
    wrist_camera: str = "15571257"
    horizon: int = 8  # actions executed per query
    max_steps: int = 600
    prefetch: int = 2  # send the next query this many steps before the chunk runs out
    noise_seed: int | None = None  # seed base; use the same value for baseline and HDH
    record_dir: str | None = None  # record rollouts for verifier training


def observe(env, args):
    obs = env.get_observation()
    images = {"external": None, "wrist": None}
    for key, img in obs["image"].items():
        if "left" in key and args.external_camera in key:
            images["external"] = img[..., :3][..., ::-1]  # BGRA -> RGB
        elif "left" in key and args.wrist_camera in key:
            images["wrist"] = img[..., :3][..., ::-1]
    assert all(v is not None for v in images.values()), f"cameras not found in {list(obs['image'])}"
    state = obs["robot_state"]
    return {"exterior": image_tools.resize_with_pad(images["external"], 224, 224),
            "wrist": image_tools.resize_with_pad(images["wrist"], 224, 224),
            "joint_position": np.asarray(state["joint_positions"]),
            "gripper_position": np.asarray([state["gripper_position"]])}


def run_episode(env, client, instruction, args, episode=None):
    pool = ThreadPoolExecutor(1)
    query_index, chunk, done_in_chunk, pending, query_id, last = 0, None, 0, None, None, None

    def request(o):
        nonlocal query_index
        req = {"observation/exterior_image_1_left": o["exterior"], "observation/wrist_image_left": o["wrist"],
               "observation/joint_position": o["joint_position"], "observation/gripper_position": o["gripper_position"],
               "prompt": instruction, "hdh/horizon": args.horizon, "hdh/query_index": query_index}
        if args.noise_seed is not None:
            req["hdh/noise_seed"] = args.noise_seed + query_index
        query_index += 1
        return o, pool.submit(client.infer, req)

    try:
        for _ in range(args.max_steps):
            t0 = time.time()
            o = observe(env, args)
            if chunk is None or done_in_chunk >= args.horizon:
                o_query, future = pending or request(o)
                result, pending, done_in_chunk = future.result(), None, 0
                chunk = np.asarray(result["actions"])
                assert chunk.shape[0] >= args.horizon and chunk.shape[1] == 8, chunk.shape
                if "hdh/selector" in result:
                    print(f"  {result['hdh/selector']}")
                if episode:
                    query_id = episode.query(o_query["exterior"], o_query["wrist"], o_query["joint_position"],
                                             o_query["gripper_position"], chunk, args.horizon)
            if args.prefetch and pending is None and args.horizon - done_in_chunk == args.prefetch:
                pending = request(o)  # overlap the round-trip with the last `prefetch` actions

            action = chunk[done_in_chunk].copy()
            action[-1] = float(action[-1] > 0.5)  # binarize gripper
            action = np.clip(action, -1, 1)
            if last is not None:
                time.sleep(max(0.0, last + 1 / HZ - time.time()))
            last = time.time()
            env.step(action)
            if episode:
                episode.step(query_id, done_in_chunk, action)
            done_in_chunk += 1
            time.sleep(max(0.0, 1 / HZ - (time.time() - t0)))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def main(args: Args):
    from droid.robot_env import RobotEnv

    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    client = WebsocketClientPolicy(args.host, args.port)
    while True:
        instruction = input("Instruction: ")
        episode = Episode(args.record_dir, instruction) if args.record_dir else None
        try:
            run_episode(env, client, instruction, args, episode)
        except KeyboardInterrupt:
            print("stopped")
        if episode:
            outcome = ""
            while outcome not in OUTCOMES:
                outcome = input(f"Outcome {OUTCOMES}: ").strip()
            episode.close(outcome)
        if input("Another episode? (y/n) ").lower() != "y":
            break
        env.reset()


if __name__ == "__main__":
    main(tyro.cli(Args))
