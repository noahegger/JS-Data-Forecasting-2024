import datetime as dt

import correlation as correlation
import feature_factory as ff
import matplotlib.dates as mdates  # Import dates directly from matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from preprocessing import Preprocessor


def apply_custom_style():
    plt.rcParams.update(
        {
            "axes.facecolor": "white",  # Changed background color to white
            "grid.color": "white",
            "grid.linestyle": "--",  # Changed gridlines to --
            "grid.linewidth": 0.7,
            "lines.linewidth": 0.5,  # Thinner lines
            "font.size": 12,
            "font.family": "monospace",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    sns.set_palette("hls")  # Changed to hls color palette


def apply_custom_style_decorator(func):
    def wrapper(*args, **kwargs):
        apply_custom_style()
        result = func(*args, **kwargs)
        plt.tight_layout()
        plt.show()
        return result

    return wrapper


@apply_custom_style_decorator
def plot_time_series(
    df, columns, title, time_column=None, start_date=None, end_date=None
):
    if start_date is not None and end_date is not None:
        df = df[(df["date_id"] >= start_date) & (df["date_id"] <= end_date)]

    plt.figure(figsize=(12, 8))
    x_axis = df[time_column] if time_column else df.index
    if isinstance(columns, list):
        for column in columns:
            plt.plot(x_axis, df[column], label=column)
        plt.legend()
    else:
        plt.plot(x_axis, df[columns], label=columns)
        plt.legend()
    plt.xlabel(time_column if time_column else "Index")
    plt.ylabel("Range")
    plt.title(title)
    plt.grid(color="lightgrey", linestyle="-", linewidth=0.5, alpha=1, zorder=0)


def get_static_correlation_matrix(df, feature_set):
    columns = feature_set + ["responder_6"]
    corr_matrix = df[columns].corr()

    plt.figure(figsize=(12, 8))  # Reduced the size of the plot
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Static Correlation Matrix")
    plt.show()

    return corr_matrix


def plot_feature_presence(df):
    feature_set = [col for col in df.columns if "feature" in col]
    presence_matrix = df[feature_set].notnull().astype(int)

    plt.figure(figsize=(12, 8))
    sns.heatmap(df[feature_set].isnull())

    plt.title("Feature Presence")
    plt.xlabel("Index")
    plt.ylabel("Feature")

    plt.show()

    return presence_matrix


def plot_lag_correlation_time_series(
    df,
    feature_set,
    responder_column="responder_6",
    lag=1,
    period=15,
    include_responder=True,
    start_date=None,
    end_date=None,
):
    if start_date is not None and end_date is not None:
        df = df[(df["date_id"] >= start_date) & (df["date_id"] <= end_date)]

    if include_responder:
        feature_set.append(responder_column)
    rolling_corr = (
        df[feature_set].rolling(window=period).corr(df[responder_column].shift(lag))
    )

    plt.figure(figsize=(12, 8))
    for feature in feature_set:
        plt.plot(rolling_corr.index, rolling_corr[feature], label=feature)
    plt.title(
        f"{period}-Period Lag {lag} Correlation with {responder_column} Over Time"
    )
    plt.xlabel("Time")
    plt.ylabel("Correlation")
    plt.grid(color="lightgrey", linestyle="-", linewidth=0.5, alpha=1, zorder=0)
    plt.legend()
    plt.show()

    return rolling_corr


def plot_mean_by_time_id(df, lookback: int, tdate: int, responder_column="responder_6"):
    lookback_dates = range(tdate - lookback, tdate)
    time_ids = df[df["date_id"] == tdate]["time_id"].unique()

    mean_values = []
    for time_id in time_ids:
        values = []
        for date in lookback_dates:
            value = df[(df["date_id"] == date) & (df["time_id"] == time_id)][
                responder_column
            ]
            if not value.empty:
                values.append(value.values[0])
        if values:
            mean_values.append(sum(values) / len(values))
        else:
            mean_values.append(None)

    mean_df = pd.DataFrame(
        {"time_id": time_ids, f"mean_{responder_column}": mean_values}
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(
        mean_df["time_id"],
        mean_df[f"mean_{responder_column}"],
        label=f"Mean {responder_column} over {lookback} days",
    )

    # Overplot responder_6 on tdate
    tdate_values = df[df["date_id"] == tdate][["time_id", responder_column]]
    ax1.plot(
        tdate_values["time_id"],
        tdate_values[responder_column],
        label=f"{responder_column} on date {tdate}",
    )

    # Add 15-period EWMA of responder_6 on tdate
    ewma_responder = tdate_values[responder_column].ewm(span=15).mean()
    ax1.plot(
        tdate_values["time_id"],
        ewma_responder,
        label=f"15-period EWMA of {responder_column} on date {tdate}",
    )

    ax1.set_title(f"Mean {responder_column} by Time ID over {lookback} Days")
    ax1.set_xlabel("Time ID")
    ax1.set_ylabel(f"Mean {responder_column}")
    ax1.grid(color="lightgrey", linestyle="-", linewidth=0.5, alpha=1, zorder=0)
    ax1.legend()

    # Plot cumulative differences
    mean_responder = tdate_values[responder_column].mean()
    cumulative_diff_mean = (tdate_values[responder_column] - mean_responder).cumsum()
    cumulative_diff_ewma = (tdate_values[responder_column] - ewma_responder).cumsum()

    ax2.plot(
        tdate_values["time_id"],
        cumulative_diff_mean,
        label=f"Cumulative Diff (Mean) {responder_column} on date {tdate}",
    )
    ax2.plot(
        tdate_values["time_id"],
        cumulative_diff_ewma,
        label=f"Cumulative Diff (EWMA) {responder_column} on date {tdate}",
    )

    ax2.set_title(f"Cumulative Differences for {responder_column} on date {tdate}")
    ax2.set_xlabel("Time ID")
    ax2.set_ylabel("Cumulative Difference")
    ax2.grid(color="lightgrey", linestyle="-", linewidth=0.5, alpha=1, zorder=0)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return mean_df


def plot_daily_responder_vs_feature(
    df, lookback: int, tdate: int, responder_column="responder_6"
):
    plt.figure(figsize=(12, 8))

    # Plot responder mean for each day in the lookback period
    lookback_dates = range(tdate - lookback, tdate + 1)
    mean_values = []
    feature_values = []
    for date in lookback_dates:
        daily_mean = df[df["date_id"] == date][responder_column].mean()
        mean_values.append(daily_mean)
        feature_values.append(
            ff.ExpWeightedMeanCalculator().calculate(df, date, responder_column)
        )

    plt.plot(
        lookback_dates,
        mean_values,
        marker="o",
        label=f"Mean {responder_column} over {lookback} days",
    )
    plt.plot(
        lookback_dates,
        feature_values,
        marker="x",
        label=f"Mean {responder_column} over {lookback} days",
    )

    # Plot exponential weighted mean for tdate

    plt.title(f"{responder_column} Mean and Feature Over Lookback {lookback} Days")
    plt.xlabel("Date ID")
    plt.ylabel(responder_column)
    plt.legend()
    plt.show()


@apply_custom_style_decorator
def plot_average_responder_over_days(
    df, lookback: int, tdate: int, responder_column="responder_6"
):
    lookback_dates = range(tdate - lookback, tdate)
    average_values = []

    for date in lookback_dates:
        daily_avg = df[df["date_id"] == date][responder_column].mean()
        average_values.append(daily_avg)

    plt.figure(figsize=(12, 8))
    plt.plot(
        lookback_dates,
        average_values,
        marker="o",
        label=f"Average {responder_column} over {lookback} days",
    )
    plt.title(f"Average {responder_column} Over {lookback} Days")
    plt.xlabel("Date ID")
    plt.ylabel(f"Average {responder_column}")
    plt.grid(color="lightgrey", linestyle="-", linewidth=0.5, alpha=1, zorder=0)
    plt.legend()
    plt.show()

    return pd.DataFrame(
        {"date_id": lookback_dates, f"average_{responder_column}": average_values}
    )


if __name__ == "__main__":

    sym = 0
    responder = 6
    partition = 0
    feature_set = [f"feature_{i:02}" for i in range(40)]  # Limit the number of features
    # feature_set = [f"feature_12"]  # Limit the number of features

    df = Preprocessor(
        symbol_id=sym,
        responder=responder,
        partition_id=partition,
        feature_set=feature_set,
        sample_frequency=5,
    ).read_partition(read_all=True)

    # plot_time_series(
    #     df[1000:2000],
    #     columns=[f"responder_{responder}"]
    #     + feature_set,  # Example with multiple columns
    #     title="Responder 6 Target Time Series",
    # )

    # get_static_correlation_matrix(df, feature_set)
    # get_rolling_correlations(df, feature_set=feature_set, window=60)

    # daily_corr_df = correlation.get_daily_correlations(df, responder, feature_set)
    # plot_time_series(
    #     df[df["date_id"] == 100],
    #     columns=["responder_6", "weight", "feature_05", "feature_06", "feature_07"],
    #     title="Responder 6 Time Series",
    #     time_column="time_id",
    #     # time_column="time_id",  # Specify the time column here
    # )

    # plot_time_series(
    #     df,
    #     columns=["responder_6", "feature_10", "feature_12"],
    #     title="Responder 6 Time Series",
    #     time_column="time_index",
    #     start_date=95,
    #     end_date=100,
    # )

    # plot_mean_by_time_id(df, lookback=15, tdate=96)
    # plot_average_responder_over_days(df, lookback=30, tdate=100)
    plot_daily_responder_vs_feature(df, lookback=150, tdate=200)
    # plot_time_series(
    #     df,
    #     columns=["responder_6", "feature_10", "feature_12"],
    #     title="Responder 6 Time Series",
    #     time_column="time_index",
    #     start_date=95,
    #     end_date=100,
    # )

    # plot_lag_correlation_time_series(
    #     df, feature_set, lag=1, period=15, start_date=95, end_date=98
    # )
    # plot_feature_presence(df[df["date_id"] == 90])
    # plot_mean_by_time_id(df, lookback=5, tdate=100)
    # plot_responder_over_time(df, lookback=15, tdate=100)
    # ff.ExpWeightedMeanCalculator().calculate(df, 100, "responder_6")
