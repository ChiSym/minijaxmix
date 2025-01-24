import jax
jax.config.update("jax_compilation_cache_dir", "jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import jax.numpy as jnp
from functools import partial
import numpy as np
from jaxtyping import Array, Float, Bool, Integer
import time

ALPHA = 1.0

def entropy(data):
    w = jnp.log(jnp.mean(data, axis=0))
    return -jnp.mean(jnp.sum(jnp.where(data, w, 0), axis=1))

def conditional_entropy(data, c):
    w = jnp.einsum('nc,nk->ck', c, data) / jnp.sum(c, axis=0)[..., None]
    p_x_y = jnp.sum(jnp.where(data[:, None, :], jnp.log(w)[None, ...], 0), axis=-1)
    res = - jnp.sum(jnp.where(c, c * p_x_y, 0), axis=0) / jnp.sum(c, axis=0)
    return res

@partial(jax.jit, static_argnames=("n_splits", "n_gibbs", "n_categories", "n_branch", "rejuvenation", "minibatch_size"))
def infer(key, data, n_splits, n_gibbs, categorical_idxs, n_categories, n_branch=2, rejuvenation=True, minibatch_size=1000):
    N = data.shape[0]
    p_ys = jnp.array([1.])
    ws = jnp.array([jnp.mean(data, axis=0)])
    assignments = jnp.zeros(N, dtype=jnp.int32)
    conditional_entropies = jnp.array([entropy(data)])

    def infer_step(step_key, minibatches, p_ys, ws, conditional_entropies):
        key1, key2 = jax.random.split(step_key)
        c = len(minibatches)
        keys = jax.random.split(key2, c)
        cluster_p_ys, cluster_ws, cluster_entropies, total_entropies = jax.vmap(split_proposal, in_axes=(0, 0, None, None, None, None))(
            keys, minibatches, n_gibbs, categorical_idxs, n_categories, n_branch)

        entropy_deltas = conditional_entropies - total_entropies
        best_idx = jnp.argmax(entropy_deltas)

        p_y = cluster_p_ys[best_idx]
        w = cluster_ws[best_idx]
        split_entropy = cluster_entropies[best_idx]

        prev_p_y = p_ys[best_idx]
        p_ys = jnp.delete(p_ys, best_idx, axis=0, assume_unique_indices=True)
        p_ys = jnp.concatenate((p_ys, prev_p_y * p_y), axis=0)

        ws = jnp.delete(ws, best_idx, axis=0, assume_unique_indices=True)
        ws = jnp.concatenate((ws, w), axis=0)

        conditional_entropies = jnp.delete(conditional_entropies, best_idx, axis=0, assume_unique_indices=True)
        conditional_entropies = jnp.concatenate((conditional_entropies, split_entropy), axis=0)

        logp_x_y = update_logp_x_y(data, ws)
        logp_y_x = update_logp_y_x(p_ys, logp_x_y)

        total_entropy = conditional_entropies @ p_ys - jnp.sum(p_ys * jnp.log(p_ys))
        # jax.debug.print("entropy: {total_entropy}", total_entropy=total_entropy)

        if rejuvenation:
            def rejuvenation_step(p_y_w, key):
                return gibbs_sampling(key, data, p_y_w[0], p_y_w[1], categorical_idxs, n_categories), None

            key1, key2 = jax.random.split(key1)
            keys = jax.random.split(key2, n_gibbs)
            (p_ys, ws), _ = jax.lax.scan(rejuvenation_step, (p_ys, ws), keys)
            logp_x_y = update_logp_x_y(data, ws)
            logp_y_x = update_logp_y_x(p_ys, logp_x_y)

            conditional_entropies = conditional_entropy(data, jnp.exp(logp_y_x))
            total_entropy = conditional_entropies @ p_ys - jnp.sum(p_ys * jnp.log(p_ys))
            # jax.debug.print("entropy after rejuvenation: {total_entropy}", total_entropy=total_entropy)

        # hard clustering
        key1, key2 = jax.random.split(key1)
        assignments = jax.random.categorical(key1, logp_y_x, axis=1)
        p_y_x = jax.nn.one_hot(assignments, num_classes=c + 1)
        logp_y_x = jnp.log(p_y_x)
        p_y = update_p_y(logp_y_x)
        w = update_w(key2, data, logp_y_x, categorical_idxs, n_categories)
        conditional_entropies = conditional_entropy(data, p_y_x)
        total_entropy = conditional_entropies @ p_y - jnp.sum(p_y * jnp.log(p_y))

        # jax.debug.print("entropy after sampling assignments: {total_entropy}", total_entropy=total_entropy)

        return p_y, w, conditional_entropies, assignments

    keys = jax.random.split(key, n_splits)
    # we could use lax.scan here, but at the cost of padding each step to the max number of clusters
    for i in range(n_splits):
        key = keys[i]
        key, subkey = jax.random.split(key)
        minibatches = make_minibatches(subkey, data, assignments, i + 1, minibatch_size)

        p_ys, ws, conditional_entropies, assignments = infer_step(key, minibatches, p_ys, ws, conditional_entropies)

    return p_ys, ws, conditional_entropies

def make_minibatches(key, data, c, num_clusters, minibatch_size):
    keys = jax.random.split(key, num_clusters)
    clusters = jax.vmap(jax.random.choice, in_axes=(0, None, None, None, 0))(keys, data, (minibatch_size,), True, (c[None, :] == jnp.arange(num_clusters)[:, None]) / jnp.sum(c[None, :] == jnp.arange(num_clusters)[:, None], axis=1)[:, None])
    return clusters

@partial(jax.jit, static_argnames=("n_gibbs", "n_segments", "n_categories"))
def split_proposal(key, data, n_gibbs, categorical_idxs: Integer[Array, "k"], n_categories: int, n_segments=2):
    N = data.shape[0]
    key, subkey = jax.random.split(key)
    alpha = jnp.ones(n_segments)
    keys = jax.random.split(subkey, N)
    p_y_x = jnp.log(jax.vmap(jax.random.dirichlet, in_axes=(0, None))(keys, alpha))
    p_y = update_p_y(p_y_x)
    key, subkey = jax.random.split(key)
    w = update_w(subkey, data, p_y_x, categorical_idxs, n_categories)

    def em_step(p_y_w, key):
        return gibbs_sampling(key, data, p_y_w[0], p_y_w[1], categorical_idxs, n_categories), None

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_gibbs)
    (p_y, w), _ = jax.lax.scan(em_step, (p_y, w), keys)
    # print(f"em_step took {end_time - start_time} seconds")

    logp_x_y = update_logp_x_y(data, w)
    logp_y_x = update_logp_y_x(p_y, logp_x_y)

    cluster_entropies = conditional_entropy(data, jnp.exp(logp_y_x))
    total_entropy = cluster_entropies @ p_y - jnp.sum(p_y * jnp.log(p_y))
    return p_y, w, cluster_entropies, total_entropy

def update_logp_x_y(data: Bool[Array, 'n k'], w_init: Float[Array, 'c k']):
    return jnp.sum(jnp.where(data[:, None, :], jnp.log(w_init)[None, ...], 0), axis=-1)

def update_logp_y_x(p_y: Float[Array, 'c'], logp_x_y: Float[Array, 'n c']):
    logp_y_x =  jnp.log(p_y)[None, :] + logp_x_y
    logZ = jax.nn.logsumexp(logp_y_x, axis=-1)
    logp_y_x = logp_y_x - logZ[..., None]
    return logp_y_x

def update_w(key: Array, data: Bool[Array, 'n k'], logp_y_x: Float[Array, 'n c'], categorical_idxs: Integer[Array, "k"], n_categories: int):
    counts = jnp.einsum('nc,nk->ck', jnp.exp(logp_y_x), data)
    alpha = counts + ALPHA
    keys = jax.random.split(key, alpha.shape[0])
    w = jax.vmap(sample_dirichlet, in_axes=(0, 0, None, None))(keys, alpha, categorical_idxs, n_categories)
    return w

def sample_dirichlet(key: Array, alpha: Float[Array, 'k'], categorical_idxs: Integer[Array, "k"], n_categories: int) -> Float[Array, 'k']:
    y = jax.random.gamma(key, alpha)
    y_sum = jax.ops.segment_sum(y, categorical_idxs, num_segments=n_categories, indices_are_sorted=True)
    y_sum_full = y_sum.take(categorical_idxs)
    return y / y_sum_full

def update_p_y(logp_y_x: Float[Array, 'n c']):
    p_y = jnp.mean(jnp.exp(logp_y_x), axis=0)
    return p_y

@partial(jax.jit, static_argnames=("n_categories"))
def gibbs_sampling(key: Array, data: Bool[Array, 'n k'], p_y: Float[Array, 'c'], w_init: Float[Array, 'c k'], categorical_idxs: Integer[Array, "k"], n_categories: int):
    logp_x_y = update_logp_x_y(data, w_init)
    logp_y_x = update_logp_y_x(p_y, logp_x_y)
    w = update_w(key, data, logp_y_x, categorical_idxs, n_categories)
    p_y = update_p_y(logp_y_x)
    return p_y, w
