import polars as pl
import numpy as np
import jax
import jax.random as random

def discretize_dataframe(df: pl.DataFrame, n_bins: int = 20):
    schema = make_schema(df)
    categorical_df = df.select(schema["types"]["categorical"])
    numerical_df = df.select(schema["types"]["piecewise_uniform"]).with_columns(
        pl.all().qcut(quantiles=n_bins).name.keep()
    )

    categorical_idxs = np.concatenate(
        [idx * np.ones(len(schema["var_metadata"][col]["levels"])) for idx, col in enumerate(categorical_df.columns)] +
        [len(categorical_df.columns) + idx * np.ones(n_bins) for idx, col in enumerate(numerical_df.columns)]
    ).astype(np.int32)

    df = pl.concat((categorical_df, numerical_df), how="horizontal")
    return schema, df, categorical_idxs

def to_dummies(df: pl.DataFrame):
    return df.to_dummies().select(pl.exclude('^.*_null$'))

def load_huggingface(dataset_path):
    splits = {
        "train": f"{dataset_path}/data-train-num.parquet",
        "test": f"{dataset_path}/data-test-full-num.parquet"
    }
    train_df = pl.read_parquet(f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['train']}")
    test_df = pl.read_parquet(f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['test']}")

    return train_df, test_df

def load_local_dataset(dataset_path):
    splits = {
        "train": f"{dataset_path}/data-train-num.parquet",
        "test": f"{dataset_path}/data-test-full-num.parquet"
    }
    train_df = pl.read_parquet(splits["train"])
    test_df = pl.read_parquet(splits["test"])

    return train_df, test_df

def load_bernoulli_noise(num_rows=100, num_cols=10, train_split=0.8):
    key = random.key(0)
    keys = random.split(key, num_rows)
    data = jax.vmap(random.bernoulli, in_axes=(0, None, None))(keys, 0.5, (num_cols,))

    train_data = data[:int(num_rows * train_split)]
    test_data = data[int(num_rows * train_split):]

    train_df = pl.DataFrame.transpose(pl.DataFrame(train_data.tolist())).cast(pl.Utf8)
    test_df = pl.DataFrame.transpose(pl.DataFrame(test_data.tolist())).cast(pl.Utf8)

    return train_df, test_df

def make_schema(df: pl.DataFrame):
    schema = {
        "types":{
            "piecewise_uniform": [],
            "normal": [],
            "categorical": []
        },
        "var_metadata":{}
    }
    for c in df.columns:
        if df[c].dtype == pl.Utf8:
            schema["types"]["categorical"].append(c)
            schema["var_metadata"][c] = {"levels": df[c].drop_nulls().unique().sort().to_list()}
        elif df[c].dtype == pl.Float64:
            schema["types"]["piecewise_uniform"].append(c)
        else:
            raise ValueError(c)
    return schema

def from_dummies(df, separator="_"):
    col_exprs = {}
    
    for col in df.columns:
        name, value = col.rsplit(separator, maxsplit=1)
        expr = pl.when(pl.col(col) == 1).then(pl.lit(value)) 
        col_exprs.setdefault(name, []).append(expr)

    return df.select(
        pl.coalesce(exprs) # keep the first non-null expression value by row
          .alias(name)
        for name, exprs in col_exprs.items()
    )
