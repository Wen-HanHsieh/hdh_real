"""openpi websocket policy server with HDH candidate selection.

    python scripts/serve.py --config pi05_droid_hdh --checkpoint <ckpt>                           # baseline
    python scripts/serve.py --config pi05_droid_hdh --checkpoint <ckpt> --selector checkpoints     # HDH, K=2
"""
import dataclasses
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from flax import nnx
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import droid_policy, policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def noise(seed, model):
    """The initial flow-matching noise pi0.5 would draw internally from this seed."""
    return jax.random.normal(jax.random.split(jax.random.key(int(seed)))[1], (1, model.action_horizon, model.action_dim))


def shared_prefix_sampler(model, num_steps=10):
    """K candidates from one prefix forward: the prefix KV cache is repeated over the K noise
    draws and the denoising loop is batched. Mirrors Pi0.sample_actions for pi0.5."""
    graphdef, state = nnx.split(model)

    @jax.jit
    def sample(state, observation, x):
        m = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)
        k, dt = x.shape[0], -1.0 / num_steps
        prefix, prefix_mask, prefix_ar = m.embed_prefix(observation)
        _, kv = m.PaliGemma.llm([prefix, None], mask=make_attn_mask(prefix_mask, prefix_ar),
                                positions=jnp.cumsum(prefix_mask, axis=1) - 1)
        kv = jax.tree.map(lambda c: jnp.repeat(c, k, axis=1), kv)
        prefix_mask = jnp.repeat(prefix_mask, k, axis=0)

        def step(carry):
            x_t, t = carry
            suffix, suffix_mask, suffix_ar, cond = m.embed_suffix(observation, x_t, jnp.broadcast_to(t, k))
            mask = jnp.concatenate([jnp.broadcast_to(prefix_mask[:, None], (k, suffix.shape[1], prefix_mask.shape[1])),
                                    make_attn_mask(suffix_mask, suffix_ar)], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, out), _ = m.PaliGemma.llm([None, suffix], mask=mask, positions=positions, kv_cache=kv, adarms_cond=[None, cond])
            return x_t + dt * m.action_out_proj(out[:, -m.action_horizon:]), t + dt

        return jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (x, 1.0))[0]

    return lambda observation, x: sample(state, observation, x)


class SelectingPolicy:
    """Draws K candidates with seeded noise, executes the one the selector prefers. Without a
    selector it is the stock policy (seeded if the client sends hdh/noise_seed)."""

    def __init__(self, policy, selector=None, k=2, shared_prefix=False):
        self.policy, self.selector, self.k = policy, selector, k
        self.model = policy._model
        self.rng = np.random.default_rng()
        self.shared = shared_prefix_sampler(self.model, policy._sample_kwargs.get("num_steps", 10)) if shared_prefix else None

    def _sample_shared(self, obs, seeds):
        inputs = self.policy._input_transform(dict(obs))
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        x = jnp.concatenate([noise(s, self.model) for s in seeds])
        actions = np.asarray(self.shared(_model.Observation.from_dict(batched), x))
        return [self.policy._output_transform({"state": np.asarray(inputs["state"]), "actions": a}) for a in actions]

    def infer(self, obs):
        seed = obs.pop("hdh/noise_seed", None)
        horizon = obs.pop("hdh/horizon", 8)
        query = obs.pop("hdh/query_index", None)
        if self.selector is None:
            return self.policy.infer(obs) if seed is None else self.policy.infer(obs, noise=noise(seed, self.model))
        t0 = time.monotonic()
        base = int(seed) if seed is not None else int(self.rng.integers(2 ** 31 - 1))
        seeds = [base + i * 1_000_003 for i in range(self.k)]  # seeds[0] is the K=1 baseline's draw
        if self.shared is not None:
            results = self._sample_shared(obs, seeds)
        else:
            results = [self.policy.infer(dict(obs), noise=noise(s, self.model)) for s in seeds]
        choice = self.selector.select(obs, [r["actions"] for r in results], int(horizon), query)
        k, record = choice if choice is not None else (0, {"task": None})
        results[k]["hdh/selector"] = {**record, "ms": 1e3 * (time.monotonic() - t0)}
        return results[k]

    def __getattr__(self, name):
        return getattr(self.policy, name)


@dataclasses.dataclass
class Args:
    config: str
    checkpoint: str
    selector: str | None = None  # directory with {plate,drawer,closet}.pt; None serves the baseline
    k: int = 2
    device: str = "cuda"
    shared_prefix: bool = False  # batch the K candidates through one prefix forward
    port: int = 8000


def main(args: Args):
    logging.basicConfig(level=logging.INFO, force=True)
    policy = policy_config.create_trained_policy(_config.get_config(args.config), args.checkpoint)
    selector = None
    if args.selector:
        from hdh.selector import Selector
        from hdh.tokens import prefix_tokens

        selector = Selector(args.selector, prefix_tokens(policy), args.device)
    served = SelectingPolicy(policy, selector, args.k, args.shared_prefix)
    # Compile before accepting connections; a cold first query outlasts the client keepalive.
    example = droid_policy.make_droid_example() | {"prompt": "put the plushie on the plate"}
    served.infer(example)
    logging.info("serving %s on port %d (%s)", args.checkpoint, args.port, "HDH" if selector else "baseline")
    websocket_policy_server.WebsocketPolicyServer(policy=served, host="0.0.0.0", port=args.port,
                                                  metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
