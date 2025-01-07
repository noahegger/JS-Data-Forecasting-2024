import datetime as dt
from typing import Optional

import calculators as calculators
import correlation as correlation
import matplotlib.dates as mdates  # Import dates directly from matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from cycler import cycler
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
    plot_scatter,
    plot_time_series,
)

from scripts.data_preprocessing import Preprocessor

# plt.style.use("dark_background")  # Changed to dark background


if __name__ == "__main__":

    start = 0
    end = 50

    data = Preprocessor(
        partition_ids=[0],
        sample_frequency=15,
    ).read_partition()
