import jax
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

def infer(key, data, n_splits, n_gibbs, categorical_idxs, n_branch=2, rejuvenation=True, minibatch_size=1000):
    n_categories = int(categorical_idxs.max() + 1)
    clusters = [data]
    p_ys = [1.]
    ws = [jnp.mean(data, axis=0)]
    entropies = [entropy(data)]
    print("initial entropy: ", entropies[0])

    for i in range(n_splits):
        key, subkey = jax.random.split(key)

        keys = jax.random.split(subkey, len(clusters))
        minibatches = jnp.array([jax.random.choice(k, cluster, shape=(minibatch_size,)) for k, cluster in zip(keys, clusters)])

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, len(clusters))
        cluster_p_ys, cluster_ws, cluster_entropies, total_entropies = jax.vmap(split_proposal, in_axes=(0, 0, None, None, None, None))(keys, minibatches, n_gibbs, categorical_idxs, n_categories, n_branch)
        deltas = [old_entropy - new_entropy for old_entropy, new_entropy in zip(entropies, total_entropies)]
        best_idx = np.argmax(deltas)

        p_y = cluster_p_ys[best_idx]
        w = cluster_ws[best_idx]

        cluster_to_split = clusters.pop(best_idx)
        p_y_x = update_p_y_x(cluster_to_split, p_y, w)
        c = jax.random.categorical(key, p_y_x, axis=1)
        clusters.extend(split_data(cluster_to_split, c))

        entropies.pop(best_idx)
        entropies.extend(cluster_entropies[best_idx])

        prev_p_y = p_ys.pop(best_idx)
        p_ys.extend(prev_p_y * cluster_p_ys[best_idx])

        ws.pop(best_idx)
        ws.extend(cluster_ws[best_idx])


        if rejuvenation:
            key, subkey = jax.random.split(key)
            concat_data = jnp.concatenate(clusters, axis=0)
            keys = jax.random.split(subkey, len(clusters))
            c = jnp.concatenate([k * jnp.ones(cluster.shape[0]).astype(jnp.int32) for k, cluster in enumerate(clusters)])
            p_y_x = jax.nn.one_hot(c, len(clusters))

            conditional_entropies = conditional_entropy(concat_data, p_y_x)
            p_y = jnp.mean(p_y_x, axis=0)
            total_entropy = conditional_entropies @ p_y - jnp.sum(p_y * jnp.log(p_y))
            print("entropy after split: ", total_entropy)

            def rejuvenation_step(p_y_w, key):
                return gibbs_sampling(key, concat_data, p_y_w[0], p_y_w[1], categorical_idxs, n_categories), None

            full_ws = jnp.stack(ws)
            full_p_ys = jnp.stack(p_ys)

            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, n_gibbs)
            (p_y_gibbs, w_gibbs), _ = jax.lax.scan(rejuvenation_step, (full_p_ys, full_ws), keys)
            p_y_x = update_p_y_x(concat_data, p_y_gibbs, w_gibbs)
            c = jax.random.categorical(key, p_y_x, axis=1)
            one_hot_c = jax.nn.one_hot(c, 2+i)
            conditional_entropies = conditional_entropy(concat_data, one_hot_c)
            total_entropy = conditional_entropies @ p_y_gibbs - jnp.sum(p_y_gibbs * jnp.log(p_y_gibbs))
            print("entropy after rejuvenation: ", total_entropy)

            clusters = split_data(concat_data, c)
            entropies = list(conditional_entropies)

    return clusters, entropies

def split_data(data, c):
    return [data[jnp.argwhere(c == k)][:, 0, :] for k in range(c.max() + 1)]

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
    start_time = time.time()
    (p_y, w), _ = jax.lax.scan(em_step, (p_y, w), keys)
    end_time = time.time()
    print(f"em_step took {end_time - start_time} seconds")

    p_y_x = update_p_y_x(data, p_y, w)

    start_time = time.time()
    cluster_entropies = conditional_entropy(data, p_y_x)
    total_entropy = cluster_entropies @ p_y - jnp.sum(p_y * jnp.log(p_y))
    end_time = time.time()
    print(f"entropy took {end_time - start_time} seconds")
    return p_y, w, cluster_entropies, total_entropy

def update_p_y_x(data: Bool[Array, 'n k'], p_y: Float[Array, 'c'], w_init: Float[Array, 'c k']):
    p_y_x =  jnp.log(p_y)[None, :] + jnp.sum(jnp.where(data[:, None, :], jnp.log(w_init)[None, ...], 0), axis=-1)
    logZ = jax.nn.logsumexp(p_y_x, axis=-1)
    p_y_x = p_y_x - logZ[..., None]
    return p_y_x

def update_w(key: Array, data: Bool[Array, 'n k'], p_y_x: Float[Array, 'n c'], categorical_idxs: Integer[Array, "k"], n_categories: int):
    counts = jnp.einsum('nc,nk->ck', jnp.exp(p_y_x), data) # / jnp.sum(jnp.exp(p_y_x), axis=0)[..., None]
    alpha = counts + ALPHA
    keys = jax.random.split(key, alpha.shape[0])
    w = jax.vmap(sample_dirichlet, in_axes=(0, 0, None, None))(keys, alpha, categorical_idxs, n_categories)
    return w

def sample_dirichlet(key: Array, alpha: Float[Array, 'k'], categorical_idxs: Integer[Array, "k"], n_categories: int) -> Float[Array, 'k']:
    y = jax.random.gamma(key, alpha)
    y_sum = jax.ops.segment_sum(y, categorical_idxs, num_segments=n_categories)
    y_sum_full = y_sum.take(categorical_idxs)
    return y / y_sum_full


def update_p_y(p_y_x: Float[Array, 'n c']):
    p_y = jnp.mean(jnp.exp(p_y_x), axis=0)
    return p_y

@partial(jax.jit, static_argnames=("n_categories"))
def gibbs_sampling(key: Array, data: Bool[Array, 'n k'], p_y: Float[Array, 'c'], w_init: Float[Array, 'c k'], categorical_idxs: Integer[Array, "k"], n_categories: int):
    p_y_x = update_p_y_x(data, p_y, w_init)
    w = update_w(key, data, p_y_x, categorical_idxs, n_categories)
    p_y = update_p_y(p_y_x)
    return p_y, w