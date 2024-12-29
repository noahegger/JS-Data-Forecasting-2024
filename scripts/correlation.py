import datetime as dt

import matplotlib.dates as mdates  # Import dates directly from matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn
from preprocessing import Preprocessor


def get_daily_correlations(df, responder, feature_set):
    daily_correlations = []
    for day in df["date_id"].unique():
        day_data = df[df["date_id"] == day]
        corr_matrix = day_data[feature_set + [f"responder_{responder}"]].corr()
        daily_correlations.append(
            corr_matrix[f"responder_{responder}"].drop(f"responder_{responder}")
        )

    daily_corr_df = pd.DataFrame(daily_correlations, index=df["date_id"].unique())
    return daily_corr_df


if __name__ == "__main__":
    feature_set = [f"feature_{i:02}" for i in range(3)]  # Limit the number of features
    symbol_id = 1
    responder = 6
    partition_id = 0
    raw_data = Preprocessor(
        symbol_id, responder, partition_id, feature_set
    ).read_partition()
