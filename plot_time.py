# %%
import plotnine as pn
import polars as pl

# %%
df = pl.read_parquet("new_distance_synth_data.parquet")
df

# %%
median_df = df.group_by(["dataset", "model", "time"]).agg(pl.median("distance").alias("median_distance"))
median_df


# %%
dataset_map = {
    "data/CTGAN/covertype": "Covertype",
    "data/CTGAN/kddcup": "KDDCup",
    "data/CTGAN/sydt": "SYDT",
    "data/lpm/CES": "CES",
    "data/lpm/PUMS": "PUMS",
    "data/lpm/PUMD": "PUMD",
}
median_df = median_df.with_columns(pl.col("dataset").replace(dataset_map))

# %%
(
    # pn.ggplot(median_df.filter(pl.col("dataset") == "data/CTGAN/covertype"))
    pn.ggplot(median_df)
    + pn.geom_line(pn.aes(x="time", y="median_distance", color="model", fill="model"))
    + pn.geom_point(pn.aes(x="time", y="median_distance", color="model", fill="model"))
    + pn.labs(y="2D Jensen-Shannon distance between\nreal and synthetic data (median)", x="Training time (seconds)")
    + pn.scale_x_log10()
    + pn.scale_y_log10()
    + pn.facet_wrap("~dataset", scales="free")
)

# %%
