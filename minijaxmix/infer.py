import jax
import jax.numpy as jnp
from functools import partial
from jaxtyping import Array, Float, Bool, Integer
from minijaxmix.query import sample_dirichlet, logprob

ALPHA_PI = 1
ALPHA_W = 1e-5

def stratified_resampling(key, w, N) :
  u = (jnp.arange(N) + jax.random.uniform(key, (N, ))) / N
  bins = jnp.cumsum(w)
  return jnp.digitize(u, bins)

@jax.named_scope("entropy")
def entropy(data):
    w = jnp.log(jnp.mean(data, axis=0))
    return -jnp.mean(jnp.sum(jnp.where(data, w, 0), axis=1))

@jax.named_scope("conditional_entropy")
def conditional_entropy(data, c):
    w = jnp.einsum('nc,nk->ck', c, data) / jnp.sum(c, axis=0)[..., None]
    p_x_y = jnp.sum(jnp.where(data[:, None, :], jnp.log(w)[None, ...], 0), axis=-1)
    res = - jnp.sum(jnp.where(c, c * p_x_y, 0), axis=0) / jnp.sum(c, axis=0)
    return res

def predictive_logprob(data, w, p_y):
    logprobs = jax.vmap(jax.vmap(logprob, in_axes=(None, 0)), in_axes=(0, None))(data, w)
    logprobs = jax.nn.logsumexp(logprobs, b=p_y, axis=1)
    logprobs = jnp.mean(logprobs)
    return logprobs

@partial(jax.jit, static_argnames=("n_iters", "proposals_per_particle", "n_gibbs", "n_categories", "rejuvenation", "minibatch_size", "test"))
@jax.named_scope("infer")
def infer(key, data, categorical_idxs, n_iters, proposals_per_particle, n_gibbs, n_categories, rejuvenation=True, minibatch_size=1000, test=False, test_data=None):
    N, k = data.shape
    n_clusters = 1 + n_iters
    p_ys = jnp.zeros(n_clusters)
    p_ys = p_ys.at[0].set(1.)
    ws = jnp.nan * jnp.zeros((n_clusters, k))
    ws = ws.at[0].set(jnp.mean(data, axis=0))
    ws = jnp.log(ws)
    conditional_H = jnp.zeros(n_clusters)
    conditional_H = conditional_H.at[0].set(entropy(data))
    total_H_rejuvenation = None

    @jax.named_scope("infer_step")
    def infer_step(carry, key_i):
        p_y, w, conditional_H = carry
        step_key, i = key_i
        logp_x_y = update_logp_x_y(data, w)
        logp_y_x = update_logp_y_x(p_y, logp_x_y)

        # choose which clusters to split
        key1, key2 = jax.random.split(step_key)
        clusters_to_split = stratified_resampling(key1, p_y, proposals_per_particle)
       
        # minibatching
        key1, key2 = jax.random.split(key1)

        logp_y_x_clusters_to_split = jax.vmap(lambda idx: logp_y_x[:, idx])(clusters_to_split).T
        minibatches = make_minibatches(key2, data, logp_y_x_clusters_to_split, minibatch_size)

        key1, key2 = jax.random.split(key1)
        keys = jax.random.split(key2, proposals_per_particle)
        cluster_p_ys, cluster_ws, cluster_H, total_H = jax.vmap(split_proposal, in_axes=(0, 0, None, None, None, None))(
            keys, minibatches, n_gibbs, categorical_idxs, n_categories, 2)

        H_deltas = p_y[clusters_to_split] * (conditional_H[clusters_to_split] - total_H)
        best_idx = jnp.nanargmax(H_deltas)
        cluster_idx = clusters_to_split[best_idx]

        best_p_y = cluster_p_ys[best_idx]
        best_w = cluster_ws[best_idx]
        best_H = cluster_H[best_idx]
        new_idxs = jnp.array([cluster_idx, i+1])

        prev_p_y = p_y[cluster_idx]
        new_p_y = prev_p_y * best_p_y
        p_y = p_y.at[new_idxs].set(new_p_y)

        w = w.at[new_idxs].set(best_w)

        conditional_H = conditional_H.at[new_idxs].set(best_H)

        logp_x_y = update_logp_x_y(data, w)
        logp_y_x = update_logp_y_x(p_y, logp_x_y)

        # total_H_split = jnp.nansum(conditional_H * p_y) - jnp.nansum(p_y * jnp.log(p_y))

        conditional_H = conditional_entropy(data, jnp.exp(logp_y_x))
        total_H_split = jnp.nansum(conditional_H * p_y) - jnp.nansum(p_y * jnp.log(p_y))

        # (p_y, w, conditional_H), total_H_rejuvenation = rejuvenation((p_y, w, conditional_H), key1)

        if test:
            logprobs = jax.vmap(jax.vmap(logprob, in_axes=(None, 0)), in_axes=(0, None))(test_data, w)
            logprobs = jax.nn.logsumexp(logprobs, b=p_y, axis=1)
            logprobs = jnp.mean(logprobs)
            return (p_y, w, conditional_H), (total_H_split, logprobs)
        else:
            return (p_y, w, conditional_H), (total_H_split, None)


    @jax.named_scope("rejuvenation")
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

    (p_ys, ws, conditional_H), (total_H_split, logprobs) = jax.lax.scan(
        infer_step, (p_ys, ws, conditional_H), (keys, jnp.arange(n_clusters - 1)))

    # if rejuvenation:
    #     (p_ys, ws, conditional_H), total_H_rejuvenation =  rejuvenation((p_ys, ws, conditional_H), key) 
    #     new_logprobs = predictive_logprob(test_data, ws, p_ys)
    #     logprobs = jnp.concatenate([logprobs, jnp.array([new_logprobs])])

    return p_ys, ws, conditional_H, total_H_split, total_H_rejuvenation, logprobs

@jax.named_scope("make_minibatches")
def make_minibatches(key, data, logp_y_x, minibatch_size):
    n_clusters = logp_y_x.shape[1]
    idxs = jax.random.categorical(key, logp_y_x[..., None], axis=0, shape=(n_clusters, minibatch_size))
    clusters = jax.vmap(jax.vmap(lambda idx: data[idx], in_axes=(0)), in_axes=(0))(idxs)
    return clusters

@partial(jax.jit, static_argnames=("n_gibbs", "n_segments", "n_categories"))
@jax.named_scope("split_proposal")
def split_proposal(key, data, n_gibbs, categorical_idxs: Integer[Array, "k"], n_categories: int, n_segments=2):
    N = data.shape[0]
    key, subkey = jax.random.split(key)
    alpha = jnp.ones(n_segments)
    keys = jax.random.split(subkey, N)
    p_y_x = jnp.log(jax.vmap(jax.random.dirichlet, in_axes=(0, None))(keys, alpha))
    p_y = update_p_y(p_y_x)
    key, subkey = jax.random.split(key)
    w = update_w(subkey, data, p_y_x, categorical_idxs, n_categories)

    @jax.profiler.annotate_function
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

@jax.named_scope("update_logp_x_y")
def update_logp_x_y(data: Bool[Array, 'n k'], w_init: Float[Array, 'c k']):
    return jnp.sum(jnp.where(data[:, None, :], w_init[None, ...], 0), axis=-1)

@jax.named_scope("update_logp_y_x")
def update_logp_y_x(p_y: Float[Array, 'c'], logp_x_y: Float[Array, 'n c']):
    logp_y_x0 =  jnp.log(p_y)[None, :] + logp_x_y
    nan_mask = jnp.isnan(logp_y_x0)
    logp_y_x1 = jnp.where(nan_mask, -jnp.inf, logp_y_x0)
    logZ = jax.nn.logsumexp(logp_y_x1, axis=-1)
    logp_y_x2 = logp_y_x1 - logZ[..., None]
    return logp_y_x2

@jax.named_scope("nanlogsumexp")
def nanlogsumexp(x):
    return jax.nn.logsumexp(jnp.where(jnp.isnan(x), 0, x))

@jax.named_scope("update_w")
def update_w(key: Array, data: Bool[Array, 'n k'], logp_y_x: Float[Array, 'n c'], categorical_idxs: Integer[Array, "k"], n_categories: int):
    p_y_x = jnp.exp(logp_y_x)
    counts = jnp.einsum('nc,nk->ck', p_y_x, data)
    # is_nan = jnp.isnan(logp_y_x).any()
    alpha = counts + ALPHA_W
    keys = jax.random.split(key, alpha.shape[0])
    w = jax.vmap(sample_dirichlet, in_axes=(0, 0, None, None))(keys, alpha, categorical_idxs, n_categories)
    return w

@jax.named_scope("update_p_y")
def update_p_y(logp_y_x: Float[Array, 'n c']):
    p_y = jnp.mean(jnp.exp(logp_y_x), axis=0)
    return p_y

@partial(jax.jit, static_argnames=("n_categories"))
@jax.named_scope("gibbs_sampling")
def gibbs_sampling(key: Array, data: Bool[Array, 'n k'], p_y: Float[Array, 'c'], w_init: Float[Array, 'c k'], categorical_idxs: Integer[Array, "k"], n_categories: int):
    logp_x_y = update_logp_x_y(data, w_init)

    logp_y_x = update_logp_y_x(p_y, logp_x_y)
    w_new = update_w(key, data, logp_y_x, categorical_idxs, n_categories)
    p_y = update_p_y(logp_y_x)

    return p_y, w_new