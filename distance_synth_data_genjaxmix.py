# # %%
# %load_ext autoreload
# %autoreload 2

# %%
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

# %%
dataset_paths = [
    "data/CTGAN/covertype", 
    "data/CTGAN/kddcup", 
    "data/CTGAN/sydt", 
    "data/lpm/CES",
    "data/lpm/PUMS",
    "data/lpm/PUMD",
]

# times = {
#     "covertype": 304.31628918647766,
#     "kddcup": 5809.921473503113,
#     "sydt": 4147.667294740677,
#     "CES": 49.699519634246826,
#     "PUMS": 557.475483417511,
#     "PUMD": 127.56089353561401,
# }
times_single_rejuvenation_100 = {
    "covertype": 5.269169092178345,
    "kddcup": 61.410168170928955,
    "sydt": 46.48610043525696,
    "CES": 2.556819438934326,
    "PUMS": 7.585843563079834,
    "PUMD": 5.1315598487854,
}
times_single_rejuvenation_300 = {
    "covertype": 35.914592266082764,
    "kddcup": 511.4724328517914,
    "sydt": 376.814204454422,
    "CES": 13.747230291366577,
    "PUMS": 57.392295598983765,
    "PUMD": 38.446903228759766,
}
times_single_rejuvenation_500 = {
    "covertype": 95.82071185112,
    "kddcup": 1392.8732736110687,
    "sydt": 1017.3137822151184,
    "CES": 33.89802026748657,
    "PUMS": 159.05469465255737,
    "PUMD": 109.2985634803772,
}



# %%
from minijaxmix.io import load_huggingface, discretize_dataframe, to_dummies
from minijaxmix.infer import sample_categorical
from minijaxmix.distances import js
from functools import partial

# partial_js = partial(js, batch_size=10)
# jit_js = jax.jit(partial_js)
jit_js = jax.jit(js)

dfs = []
for dataset_path in dataset_paths:
    print(dataset_path)
    train_df, test_df = load_huggingface(dataset_path)
    df = pl.concat((train_df, test_df))

    schema, discretized_df, categorical_idxs = discretize_dataframe(df)
    dummies_df = to_dummies(discretized_df)
    data = dummies_df.to_numpy().astype(np.bool_)

    train_data = data[:len(train_df)]
    test_data = data[len(train_df):][:10000]

    files = jnp.load(f"{dataset_path.split('/')[-1]}_single_rejuvenation.npz")

    p_ys = files["p_ys"]
    ws = files["ws"]

    n_sample = 10000

    cs = jax.random.categorical(jax.random.PRNGKey(0), jnp.log(p_ys), shape=(n_sample,))
    sample_ws = ws.take(cs, axis=0)
    n_categories = categorical_idxs.max() + 1

    samples = jax.vmap(sample_categorical, in_axes=(0, 0, None, None))(jax.random.split(jax.random.PRNGKey(0), n_sample), jnp.log(sample_ws), categorical_idxs, n_categories)

    distances = jit_js(jnp.array(test_data), jnp.array(samples))

    dfs.append(pl.DataFrame({
        "distance": np.array(distances), 
        "dataset": dataset_path, 
        "model": "GenJaxMix", 
        "time": times_single_rejuvenation_300[dataset_path.split("/")[-1]]
    }))

# %%
result_df = pl.concat(dfs)

# %%
# times_no_rejuvenation = {
#     "covertype": 35.12278389930725,
#     "kddcup": 495.74342131614685,
#     "sydt": 365.2887644767761,
#     "CES": 13.5840482711792,
#     "PUMS": 55.90764021873474,
#     "PUMD": 38.18173289299011,
# }

# %%
dfs = []
for dataset_path in dataset_paths:
    print(dataset_path)
    train_df, test_df = load_huggingface(dataset_path)
    df = pl.concat((train_df, test_df))

    schema, discretized_df, categorical_idxs = discretize_dataframe(df)
    dummies_df = to_dummies(discretized_df)
    data = dummies_df.to_numpy().astype(np.bool_)

    train_data = data[:len(train_df)]
    test_data = data[len(train_df):][:10000]

    files = jnp.load(f"{dataset_path.split('/')[-1]}_single_rejuvenation_100.npz")

    p_ys = files["p_ys"]
    ws = files["ws"]

    n_sample = 10000

    cs = jax.random.categorical(jax.random.PRNGKey(0), jnp.log(p_ys), shape=(n_sample,))
    sample_ws = ws.take(cs, axis=0)
    n_categories = categorical_idxs.max() + 1

    samples = jax.vmap(sample_categorical, in_axes=(0, 0, None, None))(jax.random.split(jax.random.PRNGKey(0), n_sample), jnp.log(sample_ws), categorical_idxs, n_categories)

    distances = jit_js(jnp.array(test_data), jnp.array(samples))

    dfs.append(pl.DataFrame({
        "distance": np.array(distances), 
        "dataset": dataset_path, 
        "model": "GenJaxMix", 
        "time": times_single_rejuvenation_100[dataset_path.split("/")[-1]]
    }))

# %%
no_rejuvenation_result_df = pl.concat(dfs)

# %%
prev_result_df = pl.read_parquet("distance_synth_data.parquet")

# %%
new_result_df = pl.concat((prev_result_df, result_df, no_rejuvenation_result_df), how="diagonal")

# %%
new_result_df.write_parquet("new_distance_synth_data.parquet")
# %%
