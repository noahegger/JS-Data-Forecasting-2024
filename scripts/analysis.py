import datetime as dt
from pathlib import Path
from typing import Optional

import calculators as calculators
import correlation as correlation
import matplotlib.dates as mdates  # Import dates directly from matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from cycler import cycler
from data_preprocessing import Preprocessor
from plotting import (
    get_static_correlation_matrix,
    plot_average_responder_over_days,
    plot_daily_responder_vs_feature,
    plot_feature_presence,
    plot_lag_correlation_time_series,
    plot_mean_by_time_id,
    plot_multi_scatter,
    plot_multiday,
    plot_multiday_with_histogram,
    plot_per_symbol_cum_error,
    plot_per_symbol_r2,
    plot_r2_time_series,
    plot_scatter,
    plot_time_series,
)

DATA_DIR = Path("/Users/noahegger/git/JS-Data-Forecasting-2024")
# plt.style.use("dark_background")  # Changed to dark background

if __name__ == "__main__":

    start = 0
    end = 1

    model_paths = [
        f"{DATA_DIR}/model_results/Lasso_1.0_r2.parquet",
        # f"{DATA_DIR}/model_results/Lasso_0.5_r2.parquet",
        # f"{DATA_DIR}/model_results/Lasso_0.1_r2.parquet",
        # f"{DATA_DIR}/model_results/Lasso_2.0_r2.parquet",
        # f"{DATA_DIR}/model_results/BaseModel_r2.parquet",
    ]

    performance_paths = [f"{DATA_DIR}/model_results/Lasso_1.0_performance.parquet"]
    symbols = [0, 1, 2, 19, 34]

    # data = Preprocessor(
    #     partition_ids=[0],
    #     sample_frequency=15,
    # ).read_partition()
    # plot_r2_time_series(model_paths, start, end)
    # plot_per_symbol_r2(performance_paths, start, end)
    plot_per_symbol_cum_error(
        performance_paths=performance_paths, start=start, end=end, symbols=symbols
    )
