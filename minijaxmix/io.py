import polars as pl
import numpy as np

def dataframe_to_arrays(df: pl.DataFrame, n_bins: int = 20):
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
    return schema, df.to_dummies().to_numpy().astype(np.bool_), categorical_idxs

def load_huggingface(dataset_path):
    splits = {
        "train": f"{dataset_path}/data-train-num.parquet",
        "test": f"{dataset_path}/data-test-full-num.parquet"
    }
    train_df = pl.read_parquet(f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['train']}")
    test_df = pl.read_parquet(f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['test']}")

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