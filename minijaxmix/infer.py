import jax
import jax.numpy as jnp
from functools import partial
from jaxtyping import Array, Float, Bool, Integer
from minijaxmix.query import sample_dirichlet, logprob

ALPHA = 1e-5

def entropy(data):
    w = jnp.log(jnp.mean(data, axis=0))
    return -jnp.mean(jnp.sum(jnp.where(data, w, 0), axis=1))

def conditional_entropy(data, c):
    w = jnp.einsum('nc,nk->ck', c, data) / jnp.sum(c, axis=0)[..., None]
    p_x_y = jnp.sum(jnp.where(data[:, None, :], jnp.log(w)[None, ...], 0), axis=-1)
    res = - jnp.sum(jnp.where(c, c * p_x_y, 0), axis=0) / jnp.sum(c, axis=0)
    return res

@partial(jax.jit, static_argnames=("n_clusters", "n_gibbs", "n_categories", "n_branch", "rejuvenation", "minibatch_size", "test"))
def infer(key, data, categorical_idxs, n_clusters, n_gibbs, n_categories, n_branch=2, rejuvenation=True, minibatch_size=1000, test=False, test_data=None):
    N, k = data.shape
    p_ys = jnp.zeros(n_clusters)
    p_ys = p_ys.at[0].set(1.)
    ws = jnp.nan * jnp.zeros((n_clusters, k))
    ws = ws.at[0].set(jnp.mean(data, axis=0))
    conditional_H = jnp.zeros(n_clusters)
    conditional_H = conditional_H.at[0].set(entropy(data))

    def infer_step(carry, key_i):
        p_y, w, conditional_H = carry
        step_key, i = key_i

        key1, key2 = jax.random.split(step_key)

        logp_x_y = update_logp_x_y(data, w)
        logp_y_x = update_logp_y_x(p_y, logp_x_y)
        
        # hard clustering
        assignments = jax.random.categorical(key2, logp_y_x, axis=1)
        p_y_x = jax.nn.one_hot(assignments, num_classes=n_clusters)
        conditional_H = conditional_entropy(data, p_y_x)
        total_H_hard_clustering = jnp.nansum(conditional_H * p_y) - jnp.nansum(p_y * jnp.log(p_y))

        key1, key2 = jax.random.split(key1)
        minibatches = make_minibatches(key2, data, assignments, n_clusters, minibatch_size)

        key1, key2 = jax.random.split(key1)
        keys = jax.random.split(key2, n_clusters)
        cluster_p_ys, cluster_ws, cluster_H, total_H = jax.vmap(split_proposal, in_axes=(0, 0, None, None, None, None))(
            keys, minibatches, n_gibbs, categorical_idxs, n_categories, n_branch)

        H_deltas = p_y * (conditional_H - total_H)
        best_idx = jnp.nanargmax(H_deltas)  # TODO: think about nans etc

        best_p_y = cluster_p_ys[best_idx]
        best_w = cluster_ws[best_idx]
        best_H = cluster_H[best_idx]

        prev_p_y = p_y[best_idx]
        p_y = p_y.at[best_idx].set(prev_p_y * best_p_y[0])
        p_y = p_y.at[i+1].set(prev_p_y * best_p_y[1])

        w = w.at[best_idx].set(best_w[0])
        w = w.at[i+1].set(best_w[1])

        conditional_H = conditional_H.at[best_idx].set(best_H[0])
        conditional_H = conditional_H.at[i+1].set(best_H[1])

        logp_x_y = update_logp_x_y(data, w)
        logp_y_x = update_logp_y_x(p_y, logp_x_y)

        total_H_split = jnp.nansum(conditional_H * p_y) - jnp.nansum(p_y * jnp.log(p_y))

        if test:
            logprobs = jax.vmap(jax.vmap(logprob, in_axes=(None, 0)), in_axes=(0, None))(test_data, w)
            logprobs = jax.nn.logsumexp(logprobs, b=p_y, axis=1)
            logprobs = jnp.sum(logprobs)
            return (p_y, w, conditional_H), (total_H_split, total_H_hard_clustering, logprobs)
        else:
            return (p_y, w, conditional_H), (total_H_split, total_H_hard_clustering, None)

    def rejuvenation(carry, key):
        p_y, w, conditional_H = carry

        def rejuvenation_step(p_y_w, key):
            return gibbs_sampling(key, data, p_y_w[0], p_y_w[1], categorical_idxs, n_categories), None

        keys = jax.random.split(key, n_gibbs)

        (p_y, w), _ = jax.lax.scan(rejuvenation_step, (p_y, w), keys)
        logp_x_y = update_logp_x_y(data, w)
        logp_y_x = update_logp_y_x(p_y, logp_x_y)

        conditional_H = conditional_entropy(data, jnp.exp(logp_y_x))
        total_H_rejuvenation = jnp.nansum(conditional_H * p_y) - jnp.nansum(p_y * jnp.log(p_y))

        return (p_y, w, conditional_H), total_H_rejuvenation

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_clusters - 1)
    # we could use lax.scan here, but at the cost of padding each step to the max number of clusters

    (p_ys, ws, conditional_H), (total_H_split, total_H_hard_clustering, logprobs) = jax.lax.scan(
        infer_step, (p_ys, ws, conditional_H), (keys, jnp.arange(n_clusters - 1)))

    if rejuvenation:
        (p_ys, ws, conditional_H), total_H_rejuvenation =  rejuvenation((p_ys, ws, conditional_H), key) 

    return p_ys, ws, conditional_H, total_H_split, total_H_rejuvenation, total_H_hard_clustering, logprobs

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

    logp_x_y = update_logp_x_y(data, w)
    logp_y_x = update_logp_y_x(p_y, logp_x_y)

    cluster_H = conditional_entropy(data, jnp.exp(logp_y_x))
    total_H = cluster_H @ p_y - jnp.sum(p_y * jnp.log(p_y))
    return p_y, w, cluster_H, total_H

def update_logp_x_y(data: Bool[Array, 'n k'], w_init: Float[Array, 'c k']):
    return jnp.sum(jnp.where(data[:, None, :], w_init[None, ...], 0), axis=-1)

def update_logp_y_x(p_y: Float[Array, 'c'], logp_x_y: Float[Array, 'n c']):
    logp_y_x0 =  jnp.log(p_y)[None, :] + logp_x_y
    nan_mask = jnp.isnan(logp_y_x0)
    logp_y_x1 = jnp.where(nan_mask, -jnp.inf, logp_y_x0)
    logZ = jax.nn.logsumexp(logp_y_x1, axis=-1)
    logp_y_x2 = logp_y_x1 - logZ[..., None]
    return logp_y_x2

def nanlogsumexp(x):
    return jax.nn.logsumexp(jnp.where(jnp.isnan(x), 0, x))

def update_w(key: Array, data: Bool[Array, 'n k'], logp_y_x: Float[Array, 'n c'], categorical_idxs: Integer[Array, "k"], n_categories: int):
    p_y_x = jnp.exp(logp_y_x)
    counts = jnp.einsum('nc,nk->ck', p_y_x, data)
    is_nan = jnp.isnan(logp_y_x).any()
    alpha = counts + ALPHA
    keys = jax.random.split(key, alpha.shape[0])
    w = jax.vmap(sample_dirichlet, in_axes=(0, 0, None, None))(keys, alpha, categorical_idxs, n_categories)
    return w

def update_p_y(logp_y_x: Float[Array, 'n c']):
    p_y = jnp.mean(jnp.exp(logp_y_x), axis=0)
    return p_y

@partial(jax.jit, static_argnames=("n_categories"))
def gibbs_sampling(key: Array, data: Bool[Array, 'n k'], p_y: Float[Array, 'c'], w_init: Float[Array, 'c k'], categorical_idxs: Integer[Array, "k"], n_categories: int):
    logp_x_y = update_logp_x_y(data, w_init)

    logp_y_x = update_logp_y_x(p_y, logp_x_y)
    w_new = update_w(key, data, logp_y_x, categorical_idxs, n_categories)
    p_y = update_p_y(logp_y_x)

    return p_y, w_new