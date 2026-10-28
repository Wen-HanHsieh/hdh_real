import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model


def prefix_tokens(policy):
    """Returns obs -> SigLIP prefix tokens (2 x 8 x 8 x 2048) of the base and left-wrist views,
    16x16 -> 8x8 by 2x2 mean pooling. One extra prefix forward; the sampler is untouched."""

    @jax.jit
    def tap(o):
        tok = policy._model.embed_prefix(o)[0][0, :512].reshape(2, 16, 16, -1)
        return tok.reshape(2, 8, 2, 8, 2, -1).mean((2, 4)).astype(jnp.bfloat16)

    def run(obs):
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], policy._input_transform(dict(obs)))
        o = _model.preprocess_observation(None, _model.Observation.from_dict(inputs), train=False)
        return np.asarray(tap(o), np.float32)

    return run
