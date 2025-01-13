import datetime as dt
from pathlib import Path
from typing import Optional

import calculators as calculators
import correlation as correlation
import matplotlib.dates as mdates  # Import dates directly from matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from cycler import cycler
from data_preprocessing import Preprocessor
from plotting import (
    get_static_correlation_matrix,
    grid_search_correlations,
    grid_search_correlations_scaled,
    grid_search_differences,
    grid_search_feature_interactions,
    grid_search_rolling_sum,
    grid_search_rolling_sum_scaled_sign,
    plot_acf_subplots,
    plot_average_responder_over_days,
    plot_cross_feature_time_series,
    plot_daily_responder_vs_feature,
    plot_day_comparison,
    plot_feature_histograms,
    plot_feature_presence,
    plot_feature_time_series,
    plot_feature_vs_responder_6_scatter,
    plot_lag_correlation_time_series,
    plot_mean_by_time_id,
    plot_multi_scatter,
    plot_multiday,
    plot_multiday_with_histogram,
    plot_per_symbol_cum_error,
    plot_per_symbol_r2,
    plot_r2_time_series,
    plot_responder_6_per_day,
    plot_scatter,
    plot_time_series,
    plot_true_vs_pred,
)

DATA_DIR = Path("/Users/noahegger/git/JS-Data-Forecasting-2024")
# plt.style.use("dark_background")  # Changed to dark background

if __name__ == "__main__":

    start = 0
    end = 1

    model_paths = [
        # f"{DATA_DIR}/model_results/Lasso_1.0_r2.parquet",
        # f"{DATA_DIR}/model_results/Lasso_1.0_r2.parquet",
        # f"{DATA_DIR}/model_results/Lasso_0.3_r2.parquet",
        # f"{DATA_DIR}/model_results/Lasso_0.5_r2.parquet",
        # f"{DATA_DIR}/model_results/Lasso_0.1_r2.parquet",
        # f"{DATA_DIR}/model_results/Lasso_2.0_r2.parquet",
        # f"{DATA_DIR}/model_results/Base15d_ExpMean_r2.parquet",
        f"{DATA_DIR}/model_results/RankedCorrelation_r2.parquet",
    ]

    performance_paths = [
        # f"{DATA_DIR}/model_results/Lasso_1.0_performance.parquet",
        # f"{DATA_DIR}/model_results/Base15d_ExpMean_performance.parquet",
        f"{DATA_DIR}/model_results/RankedCorrelation_performance.parquet",
    ]
    symbols = [0]  #
    # [0, 1, 7, 9, 10, 16, 19, 12, 13, 33]
    features = [
        "feature_05",
        "feature_06",
        "feature_07",
        "feature_08",
        "feature_09",
        "feature_10",
        "feature_11",
    ]
    full_set = [f"feature_{x:02}" for x in range(79)]
    exclude_set = [
        "feature_00",
        "feature_01",
        "feature_02",
        "feature_03",
        "feature_04",
        "feature_21",
        "feature_26",
        "feature_27",
        "feature_31",
        "feature_09",
        "feature_10",
        "feature_11",
        "feature_20",
        "feature_22",
        "feature_23",
        "feature_24",
        "feature_25",
        "feature_30",
        "feature_61",
        "feature_29",
    ]
    features = list(set(full_set) - set(exclude_set))

    # data = Preprocessor(
    #     partition_ids=[0],
    #     sample_frequency=15,
    # ).read_partition()
    # plot_r2_time_series(model_paths, start, end)
    # plot_per_symbol_r2(performance_paths, start, end)
    # plot_per_symbol_cum_error(
    #     performance_paths=performance_paths, start=start, s=e nd, symbols=symbols
    # )
    # plot_feature_time_series(
    #     performance_paths[0],
    #     start,
    #     end,
    #     16,
    #     12,
    #     ["feature_60"],  # ["feature_40" "feature_71", "feature_34", "feature_75"]
    # )
    # plot_cross_feature_time_series(
    #     performance_paths[0], start, end, 1, "feature_18", "feature_62"
    # )
    # plot_feature_histograms(performance_paths[0], 1, features)
    plot_true_vs_pred(performance_paths[0], start, end, 0)
    # plot_responder_6_per_day(performance_paths[0], start, end, 0)
    # plot_day_comparison(performance_paths[0], start, end, 1)
    # plot_feature_vs_responder_6_scatter(
    #     performance_paths[0],
    #     start,
    #     1,
    #     [
    #         "feature_34",
    #         "feature_56",
    #         "feature_35",
    #         "feature_51",
    #         "feature_48",
    #         "feature_62",
    #     ],  # ["feature_18", "feature_67"],  # ["feature_33", "feature_44", "feature_40"],
    #     transform="mean",
    # )
    # plot_acf_subplots(performance_paths[0], start, end, 1, features[10:12])
    # grid_search_correlations(
    #     performance_paths[0], symbols, features, start, end, plot_matrix=False
    # )
    # grid_search_correlations_scaled(
    #     performance_paths[0],
    #     symbols,
    #     features,
    #     start,
    #     end,
    #     cross_with="std",
    #     plot_matrix=False,
    # )
    # grid_search_feature_interactions(
    #     performance_paths[0], symbols, features, start=start, end=end
    # )
    # grid_search_rolling_sum(performance_paths[0], symbols, features, start, end)
    # grid_search_differences(performance_paths[0], symbols, features, start, end)
    # grid_search_rolling_sum_scaled_sign(
    #     performance_paths[0], symbols, features, start, end
    # )
