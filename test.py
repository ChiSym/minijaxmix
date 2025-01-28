# %%
# %load_ext autoreload
# %autoreload 2

# %%
import jax
jax.config.update("jax_compilation_cache_dir", "jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_explain_cache_misses", True)

# %%
from minijaxmix.io import load_huggingface, discretize_dataframe, to_dummies
from minijaxmix.infer import infer
import polars as pl
import jax.numpy as jnp
import time
import numpy as np
from functools import partial

dataset_paths = [
    "data/CTGAN/covertype", 
    "data/CTGAN/kddcup", 
    "data/CTGAN/sydt", 
    "data/lpm/CES",
    "data/lpm/PUMS",
    "data/lpm/PUMD",
]
for dataset_path in dataset_paths:
    print(dataset_path)
    train_df, test_df = load_huggingface(dataset_path)
    df = pl.concat((train_df, test_df))

    schema, discretized_df, categorical_idxs = discretize_dataframe(df)
    dummies_df = to_dummies(discretized_df)
    data = dummies_df.to_numpy().astype(np.bool_)

    train_data = data[:len(train_df)]
    test_data = data[len(train_df):]

    n_categories = categorical_idxs.max() + 1

    partial_infer = partial(infer, n_clusters=300, n_gibbs=20, n_categories=n_categories, n_branch=2, rejuvenation=True)
    jit_infer = jax.jit(partial_infer)

    key = jax.random.PRNGKey(0)
    start = time.time()
    lowered = jit_infer.lower(key, train_data, categorical_idxs)
    compiled = lowered.compile()

    end = time.time()
    print(f"Time compilation: {end - start}")

    start = time.time()
    p_ys, ws, conditional_entropies, total_entropies_split, total_entropies_rejuvenation, total_entropies_hard_clustering = compiled(
        key, train_data, categorical_idxs)
    end = time.time()
    print(f"Time run: {end - start}")

    dataset_name = dataset_path.split("/")[-1]
    jnp.savez(f"{dataset_name}.npz", p_ys=p_ys, ws=ws, conditional_entropies=conditional_entropies, total_entropies_split=total_entropies_split, total_entropies_rejuvenation=total_entropies_rejuvenation, total_entropies_hard_clustering=total_entropies_hard_clustering)