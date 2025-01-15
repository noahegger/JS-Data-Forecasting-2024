from typing import Optional

import calculators as calculators
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
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
            "legend.fontsize": 8,  # Smaller legend text
            "legend.title_fontsize": 10,  # Smaller legend title
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
            label=f"{file_name} Rolling Mean R² (Mean R²: {mean_r2:.5f})",
        )

    plt.xlabel("Index")
    plt.ylabel("R² Score")
    plt.title("R² Score Time Series")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


@apply_custom_style_decorator
def plot_per_symbol_r2(performance_paths, start, end):
    plt.figure(figsize=(10, 6))

    for performance_path in performance_paths:
        # Read the performance data from the Parquet file
        performance_df = pd.read_parquet(performance_path)
        performance_df = performance_df[
            (performance_df["date_id"] >= start) & (performance_df["date_id"] <= end)
        ]

        # Extract unique symbols
        symbols = performance_df["symbol_id"].unique()

        for symbol in symbols:
            symbol_df = performance_df[performance_df["symbol_id"] == symbol]
            plt.plot(symbol_df.index, symbol_df["symbol_r2"], label=f"{symbol} R²")

    plt.xlabel("Index")
    plt.ylabel("Symbol R²")
    plt.title("Symbol R² Time Series")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


@apply_custom_style_decorator
def plot_per_symbol_cum_error(
    performance_paths, start: int, end: int, symbols: Optional[list] = None
):
    plt.figure(figsize=(10, 6))

    for performance_path in performance_paths:
        # Read the performance data from the Parquet file
        performance_df = pd.read_parquet(performance_path)
        performance_df = performance_df[
            (performance_df["date_id"] >= start) & (performance_df["date_id"] <= end)
        ]

        # Extract unique symbols
        if not symbols:
            symbols = performance_df["symbol_id"].unique().tolist()

        for symbol in symbols:  # type: ignore
            symbol_df = performance_df[performance_df["symbol_id"] == symbol]
            error = (symbol_df["responder_6"] - symbol_df["responder_6_pred"]).cumsum()
            plt.plot(symbol_df.index, error, label=f"Symbol: {symbol} cum error")

    plt.xlabel("Index")
    plt.ylabel("Error")
    plt.title("Symbol Error Time Series")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


@apply_custom_style_decorator
def plot_feature_time_series(
    performance_path, start: int, end: int, symbol: int, period: int, features: list
):
    plt.figure(figsize=(10, 6))

    performance_df = pd.read_parquet(performance_path)
    performance_df = performance_df[
        (performance_df["date_id"] >= start) & (performance_df["date_id"] <= end)
    ]

    symbol_df = performance_df[performance_df["symbol_id"] == symbol]
    symbol_df = symbol_df.reset_index(drop=True)

    for feature in features:

        rolling_mean = symbol_df[feature].rolling(20).mean()

        plt.plot(
            symbol_df.index,
            # rolling_mean,
            symbol_df[feature].shift(period),  # .apply(np.sign).rolling(period).mean(),
            # * symbol_df["weight"].iloc[-1],
            # * symbol_df[feature]
            # .rolling(period)
            # .std(),  # - feature.rolling(period).mean(),
            # .rolling(5).mean(),
            # rolling_sum,  # - symbol_df[feature].shift(20),  # .rolling(20).sum(),
            label=f"{feature}",
        )

    plt.plot(symbol_df.index, symbol_df["responder_6"], label="responder_6")

    plt.xlabel("Index")
    plt.ylabel("Error")
    plt.title(f"Symbol: {symbol} Feature Time Series")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


@apply_custom_style_decorator
def plot_cross_feature_time_series(
    performance_path, start: int, end: int, symbol: int, feature1: str, feature2: str
):
    plt.figure(figsize=(10, 6))

    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)
    performance_df = performance_df[
        (performance_df["date_id"] >= start) & (performance_df["date_id"] <= end)
    ]

    # Filter the DataFrame for the given symbol
    symbol_df = performance_df[performance_df["symbol_id"] == symbol]
    symbol_df = symbol_df.reset_index(drop=True)

    # Plot the multiplication of the two features
    plt.plot(
        symbol_df.index,
        symbol_df[feature1] * symbol_df[feature2],
        label=f"{feature1} * {feature2}",
    )

    # Plot responder_6
    plt.plot(symbol_df.index, symbol_df["responder_6"], label="responder_6")

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(
        f"Time Series of {feature1} * {feature2} and responder_6 for Symbol {symbol}"
    )
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


@apply_custom_style_decorator
def plot_feature_histograms(performance_path, symbol, features):
    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)

    # Filter the DataFrame for the given symbol
    symbol_df = performance_df[performance_df["symbol_id"] == symbol]

    # Select the first 25 features
    selected_features = features[:25]

    # Create a 5x5 grid of histograms
    fig, axes = plt.subplots(5, 5, figsize=(12, 8))
    axes = axes.flatten()

    for i, feature in enumerate(selected_features):
        if feature in symbol_df.columns:
            axes[i].hist(symbol_df[feature].dropna(), bins=30, edgecolor="black")
            axes[i].set_title(feature)
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")
        else:
            axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


@apply_custom_style_decorator
def plot_true_vs_pred(performance_path, start: int, end: int, symbol: int):
    plt.figure(figsize=(10, 6))

    performance_df = pd.read_parquet(performance_path)
    performance_df = performance_df[
        (performance_df["date_id"] >= start) & (performance_df["date_id"] <= end)
    ]

    symbol_df = performance_df[performance_df["symbol_id"] == symbol]
    symbol_df = symbol_df.reset_index(drop=True)

    plt.plot(symbol_df.index, symbol_df["responder_6"], label=f"True responder_6")
    plt.plot(symbol_df.index, symbol_df["responder_6_pred"], label=f"Pred responder_6")

    plt.xlabel("Index")
    plt.ylabel("Error")
    plt.title(f"Symbol: {symbol} Prediction Time Series")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


@apply_custom_style_decorator
def plot_responder_6_per_day(performance_path, start: int, end: int, symbol: int):
    plt.figure(figsize=(10, 6))

    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)

    # Filter the DataFrame for the given symbol
    symbol_df = performance_df[performance_df["symbol_id"] == symbol]

    # Loop through each day between start and end
    for date_id in range(start, end + 1):
        # Filter the DataFrame for the current date
        date_df = symbol_df[symbol_df["date_id"] == date_id]

        # Reset the index so it starts at 0
        date_df = date_df.reset_index(drop=True)

        # Plot the "responder_6" column
        plt.plot(date_df.index, date_df["responder_6"], label=f"Date {date_id}")

    plt.xlabel("Index")
    plt.ylabel("Responder 6")
    plt.title(f"Responder 6 Time Series for Symbol {symbol}")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


@apply_custom_style_decorator
def plot_day_comparison(performance_path, day1: int, day2: int, symbol: int):
    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)

    # Filter the DataFrame for the given symbol
    symbol_df = performance_df[performance_df["symbol_id"] == symbol]

    # Filter the DataFrame for the two distinct days
    day1_df = symbol_df[symbol_df["date_id"] == day1].reset_index(drop=True)
    day2_df = symbol_df[symbol_df["date_id"] == day2].reset_index(drop=True)

    # Calculate the correlation between day1 and day2 responder_6
    min_length = min(len(day1_df), len(day2_df))
    correlation = (
        day1_df["responder_6"]
        .iloc[:min_length]
        .corr(day2_df["responder_6"].iloc[:min_length], method="pearson")
    )

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Top plot: responder_6 per day
    axes[0].plot(day1_df.index, day1_df["responder_6"], label=f"Day {day1} responder_6")
    axes[0].plot(day2_df.index, day2_df["responder_6"], label=f"Day {day2} responder_6")
    axes[0].plot(
        day2_df.index,
        day2_df["responder_6_pred"],
        label=f"Day {day2} responder_6_pred",
        linestyle="--",
    )
    average_day1_day2_pred = (day1_df["responder_6"] + day2_df["responder_6_pred"]) / 2
    axes[0].plot(
        day1_df.index,
        average_day1_day2_pred,
        label=f"Average Day {day1} responder_6 and Day {day2} responder_6_pred",
        linestyle=":",
    )
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Responder 6")
    axes[0].set_title(f"Responder 6 Time Series for Symbol {symbol}")
    axes[0].legend(loc="upper left", title=f"Correlation: {correlation:.2f}")
    axes[0].grid(True)

    # Middle plot: difference between the day_df's responder_6 values
    difference = (
        day1_df["responder_6"].iloc[:min_length]
        - day2_df["responder_6"].iloc[:min_length]
    )
    axes[1].plot(
        difference.index,
        difference,
        label=f"Difference between Day {day1} and Day {day2}",
    )
    axes[1].plot(
        day2_df.index,
        day2_df["responder_6_pred"],
        label=f"Day {day2} responder_6_pred",
        linestyle="--",
    )
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Difference in Responder 6")
    axes[1].set_title(f"Difference in Responder 6 for Symbol {symbol}")
    axes[1].legend(loc="upper left")
    axes[1].grid(True)

    # Bottom plot: cumulative sums
    cumsum_responder6_pred = (
        day2_df["responder_6"] - day2_df["responder_6_pred"]
    ).cumsum()
    cumsum_responder6_day1 = (day2_df["responder_6"] - day1_df["responder_6"]).cumsum()
    cumsum_responder6_avg = (day2_df["responder_6"] - average_day1_day2_pred).cumsum()
    axes[2].plot(
        cumsum_responder6_pred.index,
        cumsum_responder6_pred,
        label=f"Cumsum of (Day {day2} responder_6 - Day {day2} responder_6_pred)",
    )
    axes[2].plot(
        cumsum_responder6_day1.index,
        cumsum_responder6_day1,
        label=f"Cumsum of (Day {day2} responder_6 - Day {day1} responder_6)",
    )
    axes[2].plot(
        cumsum_responder6_avg.index,
        cumsum_responder6_avg,
        label=f"Cumsum of (Day {day2} responder_6 - Average Day {day1} responder_6 and Day {day2} responder_6_pred)",
    )
    axes[2].set_xlabel("Index")
    axes[2].set_ylabel("Cumulative Sum")
    axes[2].set_title(f"Cumulative Sum of Differences for Symbol {symbol}")
    axes[2].legend(loc="upper left")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


@apply_custom_style_decorator
def plot_feature_vs_responder_6_scatter(
    performance_path, day: int, symbol: int, features: list, transform=None
):
    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)

    # Filter the DataFrame for the given symbol and day
    symbol_df = performance_df[
        (performance_df["symbol_id"] == symbol) & (performance_df["date_id"] == day)
    ]

    # Select the first 9 features
    selected_features = features[:9]

    # Create a 3x3 grid of scatterplots
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for i, feature in enumerate(selected_features):
        if feature in symbol_df.columns:
            combined_df = symbol_df[[feature, "responder_6"]].copy()
            combined_df = combined_df.dropna()
            feature_values = combined_df[feature]
            responder_6_values = combined_df["responder_6"]

            if transform == "std":
                feature_values = feature_values.rolling(window=20).std()
            elif transform == "mean":
                feature_values = feature_values.rolling(window=20).mean()
            elif transform == "sign":
                feature_values = (
                    feature_values.apply(np.sign).rolling(window=20).mean()
                ) * feature_values.rolling(window=20).std()

            feature_values = feature_values.dropna()
            responder_6_values = responder_6_values.loc[feature_values.index]

            axes[i].scatter(
                feature_values,
                responder_6_values,
                edgecolor="black",
            )
            axes[i].set_title(f"{feature} ({transform}) vs. responder_6")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("responder_6")
        else:
            axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


@apply_custom_style_decorator
def plot_acf_subplots(
    performance_path, start: int, end: int, symbol: int, features: list
):
    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)

    # Filter the DataFrame for the given symbol and date range
    symbol_df = performance_df[
        (performance_df["symbol_id"] == symbol)
        & (performance_df["date_id"] >= start)
        & (performance_df["date_id"] <= end)
    ].reset_index(drop=True)

    # Select the first 2 features
    selected_features = features[:2]

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Plot ACF for responder_6
    sm.graphics.tsa.plot_acf(
        symbol_df["responder_6"].dropna(), ax=axes[0], title="ACF of responder_6"
    )

    # Plot ACF for the first feature
    if selected_features[0] in symbol_df.columns:
        sm.graphics.tsa.plot_acf(
            symbol_df[selected_features[0]].dropna(),
            ax=axes[1],
            title=f"ACF of {selected_features[0]}",
        )

    # Plot ACF for the second feature
    if len(selected_features) > 1 and selected_features[1] in symbol_df.columns:
        sm.graphics.tsa.plot_acf(
            symbol_df[selected_features[1]].dropna(),
            ax=axes[2],
            title=f"ACF of {selected_features[1]}",
        )

    plt.tight_layout()
    plt.show()


def grid_search_correlations(
    performance_path, symbols, features, start, end, plot_matrix=False
):
    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)

    results = []

    for symbol in symbols:
        # Filter the DataFrame for the given symbol
        symbol_df = performance_df[performance_df["symbol_id"] == symbol]

        # Filter the DataFrame for the given date range
        symbol_df = symbol_df[
            (symbol_df["date_id"] >= start) & (symbol_df["date_id"] <= end)
        ]

        symbol_results = []

        for feature in features:
            if feature in symbol_df.columns:
                # Compute Spearman correlation of the feature itself with responder_6
                feature_corr = symbol_df[feature].corr(
                    symbol_df["responder_6"], method="spearman"
                )
                symbol_results.append(
                    {"symbol": symbol, "feature": feature, "correlation": feature_corr}
                )

                rolling_features = {}
                for period in range(5, 21):
                    # Compute rolling mean, rolling std, absolute mean sign, and shifted feature
                    rolling_features[f"{feature}_rolling_mean_{period}"] = (
                        symbol_df[feature].rolling(window=period).mean()
                    )
                    rolling_features[f"{feature}_rolling_std_{period}"] = (
                        symbol_df[feature].rolling(window=period).std()
                    )
                    rolling_features[f"{feature}_abs_mean_sign_{period}"] = (
                        symbol_df[feature]
                        .apply(lambda x: 1 if x > 0 else -1)
                        .rolling(window=period)
                        .mean()
                        .abs()
                    )
                    rolling_features[f"{feature}_shifted_{period}"] = symbol_df[
                        feature
                    ].shift(period)

                # Concatenate all rolling features to the DataFrame at once
                rolling_df = pd.concat(rolling_features, axis=1)
                symbol_df = pd.concat([symbol_df, rolling_df], axis=1)

                for period in range(5, 21):
                    # Compute Spearman correlation with responder_6
                    rolling_mean_corr = symbol_df[
                        f"{feature}_rolling_mean_{period}"
                    ].corr(symbol_df["responder_6"], method="spearman")
                    rolling_std_corr = symbol_df[
                        f"{feature}_rolling_std_{period}"
                    ].corr(symbol_df["responder_6"], method="spearman")
                    abs_mean_sign_corr = symbol_df[
                        f"{feature}_abs_mean_sign_{period}"
                    ].corr(symbol_df["responder_6"], method="spearman")
                    shifted_corr = symbol_df[f"{feature}_shifted_{period}"].corr(
                        symbol_df["responder_6"], method="spearman"
                    )

                    # Store the results
                    symbol_results.append(
                        {
                            "symbol": symbol,
                            "feature": f"{feature}_rolling_mean_{period}",
                            "correlation": rolling_mean_corr,
                        }
                    )
                    symbol_results.append(
                        {
                            "symbol": symbol,
                            "feature": f"{feature}_rolling_std_{period}",
                            "correlation": rolling_std_corr,
                        }
                    )
                    symbol_results.append(
                        {
                            "symbol": symbol,
                            "feature": f"{feature}_abs_mean_sign_{period}",
                            "correlation": abs_mean_sign_corr,
                        }
                    )
                    symbol_results.append(
                        {
                            "symbol": symbol,
                            "feature": f"{feature}_shifted_{period}",
                            "correlation": shifted_corr,
                        }
                    )

        # Sort by the absolute value of the correlation and select the top 30
        top_features = sorted(
            symbol_results, key=lambda x: abs(x["correlation"]), reverse=True
        )[:30]

        # Print the results
        print(f"Symbol: {symbol}")
        print(f"{'Feature':<30} | {'Correlation':<10}")
        for result in top_features:
            print(f"{result['feature']:<30} | {result['correlation']:<10.4f}")

        results.extend(top_features)

        if plot_matrix:
            # Extract the top 20 features for the correlation matrix
            top_20_features = [result["feature"] for result in top_features[:20]]
            top_20_df = symbol_df[top_20_features]

            # Plot the correlation matrix
            plt.figure(figsize=(12, 10))
            corr_matrix = top_20_df.corr(method="spearman")
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
            plt.title(f"Correlation Matrix for Top 20 Features - Symbol {symbol}")
            plt.tight_layout(pad=2.0)  # Adjust padding to create bigger margins
            plt.show()

    return pd.DataFrame(results)


def grid_search_correlations_scaled(
    performance_path,
    symbols,
    features,
    start,
    end,
    cross_with="mean",
    plot_matrix=False,
):
    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)

    results = []

    for symbol in symbols:
        # Filter the DataFrame for the given symbol
        symbol_df = performance_df[performance_df["symbol_id"] == symbol]

        # Filter the DataFrame for the given date range
        symbol_df = symbol_df[
            (symbol_df["date_id"] >= start) & (symbol_df["date_id"] <= end)
        ]

        symbol_results = []

        for feature in features:
            if feature in symbol_df.columns:
                rolling_features = {}
                for period in range(20, 21):
                    # Compute rolling mean, rolling std, and absolute mean sign
                    rolling_mean = symbol_df[feature].rolling(window=period).mean()
                    rolling_std = symbol_df[feature].rolling(window=period).std()
                    rolling_med = symbol_df[feature].rolling(window=period).median()
                    mean_sign = (
                        symbol_df[feature].apply(np.sign).rolling(window=period).mean()
                    )

                    if cross_with == "mean":
                        scaled_feature = mean_sign * rolling_mean
                    elif cross_with == "std":
                        scaled_feature = mean_sign * rolling_std
                    elif cross_with == "self":
                        scaled_feature = mean_sign * symbol_df[feature]
                    elif cross_with == "median":
                        scaled_feature = mean_sign * rolling_med
                    elif cross_with == "weight":
                        scaled_feature = mean_sign * symbol_df["weight"].iloc[-1]

                    rolling_features[f"{feature}_scaled_{cross_with}_{period}"] = (
                        scaled_feature
                    )

                # Concatenate all rolling features to the DataFrame at once
                rolling_df = pd.concat(rolling_features, axis=1)
                symbol_df = pd.concat([symbol_df, rolling_df], axis=1)

                for period in range(20, 21):
                    # Compute correlation with responder_6
                    scaled_corr = symbol_df[
                        f"{feature}_scaled_{cross_with}_{period}"
                    ].corr(symbol_df["responder_6"])

                    # Store the results
                    symbol_results.append(
                        {
                            "symbol": symbol,
                            "feature": f"{feature}_scaled_{cross_with}_{period}",
                            "correlation": scaled_corr,
                        }
                    )

        # Sort by the absolute value of the correlation and select the top 30
        top_features = sorted(
            symbol_results, key=lambda x: abs(x["correlation"]), reverse=True
        )

        # Ensure only one scaled feature per feature is chosen
        chosen_features = {}
        final_top_features = []
        for result in top_features:
            base_feature = result["feature"].rsplit("_", 3)[0]
            if base_feature not in chosen_features:
                chosen_features[base_feature] = result["feature"]
                final_top_features.append(result)
            if len(final_top_features) == 30:
                break

        # Print the results
        print(f"Symbol: {symbol}")
        print(f"{'Feature':<30} | {'Correlation':<10}")
        for result in final_top_features:
            print(f"{result['feature']:<30} | {result['correlation']:<10.4f}")

        results.extend(final_top_features)

        if plot_matrix:
            # Extract the top 20 features for the correlation matrix
            top_20_features = [result["feature"] for result in final_top_features[:20]]
            top_20_df = symbol_df[top_20_features]

            # Plot the correlation matrix
            plt.figure(figsize=(12, 10))
            corr_matrix = top_20_df.corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
            plt.title(f"Correlation Matrix for Top 20 Features - Symbol {symbol}")
            plt.tight_layout(pad=2.0)  # Adjust padding to create bigger margins
            plt.show()

    return pd.DataFrame(results)


def grid_search_feature_interactions(performance_path, symbols, features, start, end):
    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)

    results = []

    for symbol in symbols:
        # Filter the DataFrame for the given symbol
        symbol_df = performance_df[performance_df["symbol_id"] == symbol]

        # Filter the DataFrame for the given date range
        symbol_df = symbol_df[
            (symbol_df["date_id"] >= start) & (symbol_df["date_id"] <= end)
        ]

        symbol_results = []

        for i, feature1 in enumerate(features):
            if feature1 in symbol_df.columns:
                for feature2 in features[i + 1 :]:
                    if feature2 in symbol_df.columns:
                        # Calculate interaction terms
                        interaction_product = symbol_df[feature1] * symbol_df[feature2]
                        interaction_division = symbol_df[feature1] / symbol_df[feature2]

                        # Compute correlation with responder_6
                        product_corr = interaction_product.corr(
                            symbol_df["responder_6"]
                        )
                        division_corr = interaction_division.corr(
                            symbol_df["responder_6"]
                        )

                        # Store the results
                        symbol_results.append(
                            {
                                "symbol": symbol,
                                "feature": f"{feature1} * {feature2}",
                                "correlation": product_corr,
                            }
                        )
                        symbol_results.append(
                            {
                                "symbol": symbol,
                                "feature": f"{feature1} / {feature2}",
                                "correlation": division_corr,
                            }
                        )

        # Sort by the absolute value of the correlation and select the top 20
        top_features = sorted(
            symbol_results, key=lambda x: abs(x["correlation"]), reverse=True
        )[:20]

        # Print the results
        print(f"Symbol: {symbol}")
        print(f"{'Feature':<30} | {'Correlation':<10}")
        for result in top_features:
            print(f"{result['feature']:<30} | {result['correlation']:<10.4f}")

        results.extend(top_features)

    return pd.DataFrame(results)


def grid_search_rolling_sum(performance_path, symbols, features, start, end):
    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)

    results = []

    for symbol in symbols:
        # Filter the DataFrame for the given symbol
        symbol_df = performance_df[performance_df["symbol_id"] == symbol]

        # Filter the DataFrame for the given date range
        symbol_df = symbol_df[
            (symbol_df["date_id"] >= start) & (symbol_df["date_id"] <= end)
        ]

        symbol_results = []

        for feature in features:
            if feature in symbol_df.columns:
                rolling_features = {}
                for period in range(5, 21):
                    # Compute rolling sum
                    rolling_sum = symbol_df[feature].rolling(window=period).sum()
                    rolling_features[f"{feature}_rolling_sum_{period}"] = rolling_sum

                # Concatenate all rolling features to the DataFrame at once
                rolling_df = pd.concat(rolling_features, axis=1)
                symbol_df = pd.concat([symbol_df, rolling_df], axis=1)

                for period in range(5, 21):
                    # Compute correlation with responder_6
                    rolling_sum_corr = symbol_df[
                        f"{feature}_rolling_sum_{period}"
                    ].corr(symbol_df["responder_6"])

                    # Store the results
                    symbol_results.append(
                        {
                            "symbol": symbol,
                            "feature": f"{feature}_rolling_sum_{period}",
                            "correlation": rolling_sum_corr,
                        }
                    )

        # Sort by the absolute value of the correlation and select the top 30
        top_features = sorted(
            symbol_results, key=lambda x: abs(x["correlation"]), reverse=True
        )[:30]

        # Print the results
        print(f"Symbol: {symbol}")
        print(f"{'Feature':<30} | {'Correlation':<10}")
        for result in top_features:
            print(f"{result['feature']:<30} | {result['correlation']:<10.4f}")

        results.extend(top_features)

    return pd.DataFrame(results)


def grid_search_differences(performance_path, symbols, features, start, end):
    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)

    results = []

    for symbol in symbols:
        # Filter the DataFrame for the given symbol
        symbol_df = performance_df[performance_df["symbol_id"] == symbol]

        # Filter the DataFrame for the given date range
        symbol_df = symbol_df[
            (symbol_df["date_id"] >= start) & (symbol_df["date_id"] <= end)
        ]

        symbol_results = []

        for feature in features:
            if feature in symbol_df.columns:
                difference_features = {}
                for period in range(5, 21):
                    # Compute differences
                    difference = symbol_df[feature] - symbol_df[feature].shift(period)
                    difference_features[f"{feature}_difference_{period}"] = difference

                # Concatenate all difference features to the DataFrame at once
                difference_df = pd.concat(difference_features, axis=1)
                symbol_df = pd.concat([symbol_df, difference_df], axis=1)

                for period in range(5, 21):
                    # Compute correlation with responder_6
                    difference_corr = symbol_df[f"{feature}_difference_{period}"].corr(
                        symbol_df["responder_6"]
                    )

                    # Store the results
                    symbol_results.append(
                        {
                            "symbol": symbol,
                            "feature": f"{feature}_difference_{period}",
                            "correlation": difference_corr,
                        }
                    )

        # Sort by the absolute value of the correlation and select the top 30
        top_features = sorted(
            symbol_results, key=lambda x: abs(x["correlation"]), reverse=True
        )[:30]

        # Print the results
        print(f"Symbol: {symbol}")
        print(f"{'Feature':<30} | {'Correlation':<10}")
        for result in top_features:
            print(f"{result['feature']:<30} | {result['correlation']:<10.4f}")

        results.extend(top_features)

    return pd.DataFrame(results)


def grid_search_rolling_sum_scaled_sign(
    performance_path, symbols, features, start, end
):
    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)

    results = []

    for symbol in symbols:
        # Filter the DataFrame for the given symbol
        symbol_df = performance_df[performance_df["symbol_id"] == symbol]

        # Filter the DataFrame for the given date range
        symbol_df = symbol_df[
            (symbol_df["date_id"] >= start) & (symbol_df["date_id"] <= end)
        ]

        symbol_results = []

        for feature in features:
            if feature in symbol_df.columns:
                rolling_features = {}
                for period in range(5, 21):
                    # Compute rolling sum and absolute mean sign
                    rolling_sum = symbol_df[feature].rolling(window=period).sum()
                    abs_mean_sign = (
                        symbol_df[feature]
                        .apply(lambda x: 1 if x > 0 else -1)
                        .rolling(window=period)
                        .mean()
                        .abs()
                    )
                    scaled_rolling_sum = rolling_sum * abs_mean_sign

                    rolling_features[f"{feature}_rolling_sum_scaled_sign_{period}"] = (
                        scaled_rolling_sum
                    )

                # Concatenate all rolling features to the DataFrame at once
                rolling_df = pd.concat(rolling_features, axis=1)
                symbol_df = pd.concat([symbol_df, rolling_df], axis=1)

                for period in range(5, 21):
                    # Compute correlation with responder_6
                    scaled_rolling_sum_corr = symbol_df[
                        f"{feature}_rolling_sum_scaled_sign_{period}"
                    ].corr(symbol_df["responder_6"])

                    # Store the results
                    symbol_results.append(
                        {
                            "symbol": symbol,
                            "feature": f"{feature}_rolling_sum_scaled_sign_{period}",
                            "correlation": scaled_rolling_sum_corr,
                        }
                    )

        # Sort by the absolute value of the correlation and select the top 30
        top_features = sorted(
            symbol_results, key=lambda x: abs(x["correlation"]), reverse=True
        )[:30]

        # Print the results
        print(f"Symbol: {symbol}")
        print(f"{'Feature':<30} | {'Correlation':<10}")
        for result in top_features:
            print(f"{result['feature']:<30} | {result['correlation']:<10.4f}")

        results.extend(top_features)

    return pd.DataFrame(results)


def grid_search_sign_correlations(
    performance_path, symbols, features, start, end, plot_matrix=False
):
    # Read the performance data from the Parquet file
    performance_df = pd.read_parquet(performance_path)

    results = []

    for symbol in symbols:
        # Filter the DataFrame for the given symbol
        symbol_df = performance_df[performance_df["symbol_id"] == symbol]

        # Filter the DataFrame for the given date range
        symbol_df = symbol_df[
            (symbol_df["date_id"] >= start) & (symbol_df["date_id"] <= end)
        ]

        symbol_results = []

        for feature in features:
            if feature in symbol_df.columns:
                rolling_features = {}
                for period in range(20, 21):
                    # Compute rolling mean, absolute mean sign, and shifted feature
                    rolling_features[f"{feature}_rolling_mean_{period}"] = (
                        symbol_df[feature].rolling(window=period).mean()
                    )
                    # rolling_features[feature] = symbol_df[feature]
                    rolling_features[f"{feature}_mean_sign_{period}"] = (
                        symbol_df[feature]
                        .apply(lambda x: 1 if x > 0 else -1)
                        .rolling(window=period)
                        .mean()
                    )
                    rolling_features[f"{feature}_shifted_{period}"] = symbol_df[
                        feature
                    ].shift(period)

                # Concatenate all rolling features to the DataFrame at once
                rolling_df = pd.concat(rolling_features, axis=1)
                symbol_df = pd.concat([symbol_df, rolling_df], axis=1)

                for period in range(20, 21):
                    # Compute sign correlation with responder_6
                    rolling_mean_sign_corr = (
                        symbol_df[f"{feature}_rolling_mean_{period}"]
                        .apply(np.sign)
                        .corr(
                            symbol_df["responder_6"].apply(np.sign), method="spearman"
                        )
                    )
                    feature_sign_corr = (
                        symbol_df[feature]
                        .apply(np.sign)
                        .corr(
                            symbol_df["responder_6"].apply(np.sign), method="spearman"
                        )
                    )
                    abs_mean_sign_corr = (
                        symbol_df[f"{feature}_mean_sign_{period}"]
                        .apply(np.sign)
                        .corr(
                            symbol_df["responder_6"].apply(np.sign), method="spearman"
                        )
                    )
                    shifted_sign_corr = (
                        symbol_df[f"{feature}_shifted_{period}"]
                        .apply(np.sign)
                        .corr(
                            symbol_df["responder_6"].apply(np.sign), method="spearman"
                        )
                    )

                    # Store the results
                    symbol_results.append(
                        {
                            "symbol": symbol,
                            "feature": f"{feature}_rolling_mean_sign_{period}",
                            "correlation": rolling_mean_sign_corr,
                        }
                    )
                    symbol_results.append(
                        {
                            "symbol": symbol,
                            "feature": feature,
                            "correlation": feature_sign_corr,
                        }
                    )
                    symbol_results.append(
                        {
                            "symbol": symbol,
                            "feature": f"{feature}_abs_mean_sign_{period}",
                            "correlation": abs_mean_sign_corr,
                        }
                    )
                    symbol_results.append(
                        {
                            "symbol": symbol,
                            "feature": f"{feature}_shifted_sign_{period}",
                            "correlation": shifted_sign_corr,
                        }
                    )

        # Sort by the absolute value of the correlation and select the top 30
        top_features = sorted(
            symbol_results, key=lambda x: abs(x["correlation"]), reverse=True
        )[:30]

        # Print the results
        print(f"Symbol: {symbol}")
        print(f"{'Feature':<30} | {'Correlation':<10}")
        for result in top_features:
            print(f"{result['feature']:<30} | {result['correlation']:<10.4f}")

        results.extend(top_features)

        if plot_matrix:
            # Extract the top 20 features for the correlation matrix
            top_20_features = [result["feature"] for result in top_features[:20]]
            top_20_df = symbol_df[top_20_features]

            # Plot the correlation matrix
            plt.figure(figsize=(12, 10))
            corr_matrix = top_20_df.corr(method="spearman")
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
            plt.title(f"Correlation Matrix for Top 20 Features - Symbol {symbol}")
            plt.tight_layout(pad=2.0)  # Adjust padding to create bigger margins
            plt.show()

    return pd.DataFrame(results)
