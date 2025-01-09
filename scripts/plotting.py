from typing import Optional

import calculators as calculators
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
import utils as utils
from calculators import ExpWeightedMeanCalculator
from cycler import cycler
from data_preprocessing import Preprocessor


def apply_custom_style():
    custom_palette = [
        "#8be9fd",
        "#ff79c6",
        "#50fa7b",
        "#bd93f9",
        "#ffb86c",
        "#ff5555",
        "#f1fa8c",
        "#6272a4",
    ]  # Custom colors from dracula.mplstyle
    plt.rcParams.update(
        {
            "lines.color": "#F8F8F2",
            "patch.edgecolor": "#F8F8F2",
            "text.color": "#F8F8F2",
            "axes.facecolor": "#282A36",  # Grey-black background
            "axes.edgecolor": "#F8F8F2",
            "axes.labelcolor": "#F8F8F2",
            "axes.prop_cycle": cycler("color", custom_palette),
            "xtick.color": "#F8F8F2",
            "ytick.color": "#F8F8F2",
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#44475A",
            "grid.color": "#F8F8F2",
            "grid.linestyle": "--",  # Changed gridlines to --
            "grid.linewidth": 0.7,
            "lines.linewidth": 0.5,  # Thinner lines
            "font.size": 12,
            "font.family": "monospace",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "#282A36",  # Grey-black background
            "figure.edgecolor": "#282A36",
            "savefig.facecolor": "#282A36",
            "savefig.edgecolor": "#282A36",
            "boxplot.boxprops.color": "#F8F8F2",
            "boxplot.capprops.color": "#F8F8F2",
            "boxplot.flierprops.color": "#F8F8F2",
            "boxplot.flierprops.markeredgecolor": "#F8F8F2",
            "boxplot.whiskerprops.color": "#F8F8F2",
        }
    )
    sns.set_palette(custom_palette)  # Set custom color palette


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
    df,
    columns,
    title,
    time_column: str,
    start_date: Optional[int],
    end_date: Optional[int],
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


@apply_custom_style_decorator
def plot_multiday(df: pd.DataFrame, columns: list[str], start_date: int):
    fig, axes = plt.subplots(3, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i in range(9):
        date_id = start_date + i
        print(date_id)
        daily_df = df[df["date_id"] == date_id]
        ax = axes[i]
        for column in columns:
            ax.plot(daily_df["time_id"], daily_df[column], label=column)
        ax.set_title(f"Date ID: {date_id}")
        ax.set_xlabel("Time ID")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(color="lightgrey", linestyle="-", linewidth=0.5, alpha=1, zorder=0)

    plt.tight_layout()
    plt.show()


@apply_custom_style_decorator
def plot_multiday_with_histogram(df: pd.DataFrame, columns: list[str], start_date: int):
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(3):
        date_id = start_date + i
        daily_df = df[df["date_id"] == date_id]

        # Time series plot
        ax_time_series = axes[i * 2]
        for column in columns:
            ax_time_series.plot(daily_df["time_id"], daily_df[column], label=column)
        ax_time_series.set_title(f"Date ID: {date_id}")
        ax_time_series.set_xlabel("Time ID")
        ax_time_series.set_ylabel("Value")
        ax_time_series.legend(loc="upper right")
        ax_time_series.grid(
            color="lightgrey", linestyle="-", linewidth=0.5, alpha=1, zorder=0
        )

        # Histogram plot
        ax_histogram = axes[i * 2 + 1]
        for column in columns:
            ax_histogram.hist(daily_df[column], bins=30, alpha=0.5, label=column)
        ax_histogram.set_title(f"Histogram for Date ID: {date_id}")
        ax_histogram.set_xlabel("Value")
        ax_histogram.set_ylabel("Frequency")
        ax_histogram.legend(loc="upper right")
        ax_histogram.grid(
            color="lightgrey", linestyle="-", linewidth=0.5, alpha=1, zorder=0
        )

    plt.tight_layout()
    plt.show()


@apply_custom_style_decorator
def get_static_correlation_matrix(df, feature_set):
    columns = feature_set + ["responder_6"]
    corr_matrix = df[columns].corr()

    plt.figure(figsize=(12, 8))  # Reduced the size of the plot
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Static Correlation Matrix")
    plt.show()

    return corr_matrix


@apply_custom_style_decorator
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


@apply_custom_style_decorator
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


@apply_custom_style_decorator
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
    ewma_responder = (
        tdate_values[responder_column].ewm(halflife=3, min_periods=15).mean()
    )
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
    mean_responder = mean_values
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


@apply_custom_style_decorator
def plot_daily_responder_vs_feature(
    df, lookback: int, tdate: int, feature: str, responder_column="responder_6"
):
    plt.figure(figsize=(12, 8))

    # Plot responder mean for each day in the lookback period
    lookback_dates = range(tdate - lookback, tdate + 1)
    mean_values = []
    feature_values = []
    for date in lookback_dates:
        daily_mean = df[df["date_id"] == date][responder_column].mean()
        mean_values.append(daily_mean)
        feature_values.append(ExpWeightedMeanCalculator().calculate(mean_values))

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
        label=f"{feature} over {lookback} days",
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


@apply_custom_style_decorator
def plot_scatter(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.DataFrame,
    title: str,
    display_regression: bool = True,
):

    # Calculate regression
    model = calculators.LinearRegressionCalculator()
    model.fit(X, y)
    r2 = model.score(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, label="Data points")
    if display_regression:
        plt.plot(X, y_pred, color="red", label=f"Regression line (R²: {r2:.2f})")
    plt.title(title)
    plt.xlabel(X.columns[0])
    plt.ylabel(y.columns[0])
    plt.legend()
    plt.grid(color="lightgrey", linestyle="-", linewidth=0.5, alpha=1, zorder=0)
    plt.show()


@apply_custom_style_decorator
def plot_multi_scatter(df: pd.DataFrame, y: pd.DataFrame, feature_columns: list[str]):
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, feature_column in enumerate(feature_columns):
        X = df[[feature_column]]

        # Calculate regression
        model = calculators.LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        y_pred = model.predict(X)

        ax = axes[i]
        ax.scatter(X, y, label="Data points")
        ax.plot(X, y_pred, color="red", label=f"Regression line (R²: {r2:.2f})")
        ax.set_title(f"{feature_column} vs. {y.columns[0]}")
        ax.set_xlabel(feature_column)
        ax.set_ylabel(y.columns[0])
        ax.legend()
        ax.grid(color="lightgrey", linestyle="-", linewidth=0.5, alpha=1, zorder=0)

    plt.tight_layout()
    plt.show()


def plot_predictions_vs_true(predictions: pd.DataFrame, true_values: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(predictions["date_id"], predictions["responder_6"], label="Predictions")
    plt.plot(true_values["date_id"], true_values["responder_6"], label="True Values")
    plt.xlabel("Date ID")
    plt.ylabel("Responder 6")
    plt.legend()
    plt.show()


@apply_custom_style_decorator
def plot_r2_time_series(r2_parquet_paths, start, end):
    plt.figure(figsize=(10, 6))

    for r2_parquet_path in r2_parquet_paths:
        # Read the R² data from the Parquet file
        r2_df = pd.read_parquet(r2_parquet_path)
        r2_df = r2_df[(r2_df["date_id"] >= start) & (r2_df["date_id"] <= end)]

        rolling_mean = r2_df["r2"].rolling(window=30).mean()
        mean_r2 = r2_df["r2"].mean()

        file_name = r2_parquet_path.split("/")[-1].split(".")[0]

        # Plot R² Score and Rolling Mean R²
        plt.plot(r2_df.index, r2_df["r2"], linestyle="--", label=f"{file_name} R²")
        plt.plot(
            r2_df.index,
            rolling_mean,
            linestyle="-",
            label=f"{file_name} Rolling Mean R² (Mean R²: {mean_r2:.2f})",
        )

    plt.xlabel("Index")
    plt.ylabel("R² Score")
    plt.title("R² Score Time Series")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()
