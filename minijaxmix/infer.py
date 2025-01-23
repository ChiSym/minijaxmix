import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from jaxtyping import Array, Float, Bool

def entropy(data):
    w = jnp.log(jnp.mean(data, axis=0))
    return -jnp.mean(jnp.sum(jnp.where(data, w, 0), axis=1))

def clustering_entropy(clusters):
    entropies = [entropy(cluster) for cluster in clusters]
    p_y = np.array([cluster.shape[0] for cluster in clusters])
    p_y = p_y / p_y.sum()
    return sum(p_y * entropies) - sum(p_y * jnp.log(p_y))

def infer(key, data, n_splits, n_gibbs, n_branch=2, rejuvenation=True):
    clusters = [data]
    entropies = [entropy(data)]
    print("initial entropy: ", entropies[0])

    for i in range(n_splits):
        key, subkey = jax.random.split(key)
        cs, cluster_entropies, total_entropies = zip(*[split_proposal(subkey, cluster, n_gibbs, n_segments=n_branch) for cluster in clusters])
        deltas = [old_entropy - new_entropy for old_entropy, new_entropy in zip(entropies, total_entropies)]
        best_idx = np.argmax(deltas)

        cluster_to_split = clusters.pop(best_idx)
        clusters.extend(split_data(cluster_to_split, cs[best_idx]))

        entropies.pop(best_idx)
        entropies.extend(cluster_entropies[best_idx])

        print("after split: ", clustering_entropy(clusters))

        # rejuvenation
        if rejuvenation:
            key, subkey = jax.random.split(key)
            concat_data = jnp.concatenate(clusters, axis=0)
            c = jnp.concatenate([k * jnp.ones(cluster.shape[0]).astype(jnp.int32) for k, cluster in enumerate(clusters)])
            p_y_x = jax.nn.one_hot(c, len(clusters))

            def rejuvenation_step(p_y_x, _):
                return gibbs_sampling(concat_data, p_y_x), None

            p_y_x, _ = jax.lax.scan(rejuvenation_step, jnp.log(p_y_x), length=n_gibbs)
            c = jax.random.categorical(key, p_y_x, axis=1)
            clusters = split_data(concat_data, c)
            entropies = [entropy(cluster) for cluster in clusters]

            print("after rejuvenation: ", clustering_entropy(clusters))

    return clusters, entropies

def split_data(data, c):
    return [data[jnp.argwhere(c == k)][:, 0, :] for k in range(c.max() + 1)]

def split_proposal(key, data, n_gibbs, n_segments=2):
    N = data.shape[0]
    key, subkey = jax.random.split(key)
    alpha = jnp.ones(n_segments)
    keys = jax.random.split(subkey, N)
    p_y_x = jnp.log(jax.vmap(jax.random.dirichlet, in_axes=(0, None))(keys, alpha))

    def em_step(p_y_x, _):
        return gibbs_sampling(data, p_y_x), None

    p_y_x, _ = jax.lax.scan(em_step, p_y_x, length=n_gibbs)
    p_y = jnp.mean(jnp.exp(p_y_x), axis=0)
    c = jax.random.categorical(key, p_y_x, axis=1)
    
    clusters = split_data(data, c)

    cluster_entropies = [entropy(cluster) for cluster in clusters]
    total_entropy = clustering_entropy(clusters)
    return c, cluster_entropies, total_entropy

def gibbs_sampling(data: Bool[Array, 'n k'], p_y_x_init: Float[Array, 'n c']):
    w = jnp.einsum('nc,nk->ck', jnp.exp(p_y_x_init), data) / jnp.sum(jnp.exp(p_y_x_init), axis=0)[..., None]
    p_y = jnp.mean(jnp.exp(p_y_x_init), axis=0)
    p_y_x =  jnp.log(p_y)[None, :] + jnp.sum(jnp.where(data[:, None, :], jnp.log(w)[None, ...], 0), axis=-1)
    logZ = jax.nn.logsumexp(p_y_x, axis=1)
    # print(jnp.mean(jnp.exp(logZ)))
    p_y_x = p_y_x - logZ[..., None]
    return p_y_x
