from jaxtyping import Array, Float, Int, Bool, Integer
from functools import partial
import jax
import jax.numpy as jnp


def logprob(obs: Bool[Array, "k"], w: Float[Array, "k"]) -> Float[Array, ""]:
    return jnp.sum(obs * w)

def condition(obs: Bool[Array, "k"], logpi: Float[Array, "c"], w: Float[Array, "c k"]):
    logp = jax.vmap(logprob, in_axes=(None, 0))(obs, w)
    posterior_logpi = logpi + logp
    return jax.nn.log_softmax(posterior_logpi)

@partial(jax.jit, static_argnames=("N", "n_categories"))
def sample(key: Array, logpi: Float[Array, "c"], w: Float[Array, "c k"], N: int, categorical_idxs: Integer[Array, "k"], n_categories: int) -> Bool[Array, "k"]:
    key, subkey = jax.random.split(key)
    idxs = jax.random.categorical(subkey, logpi, shape=(N,))
    ws = w.take(idxs, axis=0)

    keys = jax.random.split(key, N)
    return jax.vmap(sample_categorical, in_axes=(0, 0, None, None))(keys, ws, categorical_idxs, n_categories)

def sample_dirichlet(key: Array, alpha: Float[Array, 'k'], categorical_idxs: Integer[Array, "k"], n_categories: int) -> Float[Array, 'k']:
    y = jax.random.loggamma(key, alpha)
    c = jnp.max(y)
    y_exp = jnp.exp(y - c)
    y_sum = jax.ops.segment_sum(y_exp, categorical_idxs, num_segments=n_categories)
    log_y_sum = jnp.log(y_sum)
    y_sum_full = log_y_sum.take(categorical_idxs)
    y_sum_full += c

    return y - y_sum_full

def sample_categorical(key: Array, logprobs: Float[Array, 'k'], categorical_idxs: Integer[Array, "k"], n_categories: int) -> Bool[Array, 'k']:
    x = jax.random.gumbel(key, shape=logprobs.shape[0])
    maxes = jax.ops.segment_max(logprobs + x, categorical_idxs, num_segments=n_categories)
    maxes = maxes.take(categorical_idxs)
    return maxes == logprobs + x