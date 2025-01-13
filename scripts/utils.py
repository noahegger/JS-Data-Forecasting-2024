from abc import ABC, abstractmethod

import numpy as np
import polars as pl


def get_time_weights(n, halflife=0.35):
    decay_factor = 0.5 ** (1 / (halflife * n))
    weights = decay_factor ** np.arange(n)
    weights /= weights.sum()
    return weights


def time_weighted_mean(vals, halflife=0.35):
    n = len(vals)
    weights = get_time_weights(n, halflife)
    return np.dot(vals, weights[::-1])


def get_custom_r2(y_true, y_pred, weights=None):
    """
    Calculate the custom R² score.

    :param y_true: Actual values.
    :param y_pred: Predicted values.
    :param weights: Optional weights for the calculation.
    :return: R² score.
    """
    if weights is None:
        weights = np.ones_like(y_true)

    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true) ** 2)
    return 1 - ss_res / ss_tot


def get_r2_by_time_id(
    df, date_id, time_id, y_true_col, y_pred_col, weights_col=None, symbol=None
):
    """
    Calculate the custom R² score for a specific time_id across all symbols or a single symbol.

    :param df: DataFrame containing the data.
    :param date_id: Specific date_id to filter the data.
    :param time_id: Specific time_id to filter the data.
    :param y_true_col: Column name for actual values.
    :param y_pred_col: Column name for predicted values.
    :param weights_col: Optional column name for weights.
    :param symbol: Optional symbol to filter the data.
    :return: R² score.
    """
    filtered_df = df[(df["date_id"] == date_id) & (df["time_id"] == time_id)]
    if symbol:
        filtered_df = filtered_df[filtered_df["symbol_id"] == symbol]
    y_true = filtered_df[y_true_col].values
    y_pred = filtered_df[y_pred_col].values
    weights = filtered_df[weights_col].values if weights_col else None
    return get_custom_r2(y_true, y_pred, weights)


def get_r2_by_date_id(
    df, date_id, y_true_col, y_pred_col, weights_col=None, symbol=None
):
    """
    Calculate the custom R² score for a specific date_id across all symbols or a single symbol.

    :param df: DataFrame containing the data.
    :param date_id: Specific date_id to filter the data.
    :param y_true_col: Column name for actual values.
    :param y_pred_col: Column name for predicted values.
    :param weights_col: Optional column name for weights.
    :param symbol: Optional symbol to filter the data.
    :return: R² score.
    """
    filtered_df = df[df["date_id"] == date_id]
    if symbol:
        filtered_df = filtered_df[filtered_df["symbol_id"] == symbol]
    y_true = filtered_df[y_true_col].values
    y_pred = filtered_df[y_pred_col].values
    weights = filtered_df[weights_col].values if weights_col else None
    return get_custom_r2(y_true, y_pred, weights)


def get_time_series_r2(
    df, date_id, y_true_col, y_pred_col, weights_col=None, symbol=None
):
    """
    Calculate the time series of R² values for a specific date_id across all symbols or a single symbol.

    :param df: DataFrame containing the data.
    :param date_id: Specific date_id to filter the data.
    :param y_true_col: Column name for actual values.
    :param y_pred_col: Column name for predicted values.
    :param weights_col: Optional column name for weights.
    :param symbol: Optional symbol to filter the data.
    :return: Series of R² values.
    """
    time_ids = df[df["date_id"] == date_id]["time_id"].unique()
    r2_series = {}
    for time_id in time_ids:
        r2_series[time_id] = get_r2_by_time_id(
            df, date_id, time_id, y_true_col, y_pred_col, weights_col, symbol
        )
    return r2_series


def get_date_series_r2(df, y_true_col, y_pred_col, weights_col=None, symbol=None):
    """
    Calculate the date series of R² values across all symbols or a single symbol.

    :param df: DataFrame containing the data.
    :param y_true_col: Column name for actual values.
    :param y_pred_col: Column name for predicted values.
    :param weights_col: Optional column name for weights.
    :param symbol: Optional symbol to filter the data.
    :return: Series of R² values.
    """
    date_ids = df["date_id"].unique()
    r2_series = {}
    for date_id in date_ids:
        r2_series[date_id] = get_r2_by_date_id(
            df, date_id, y_true_col, y_pred_col, weights_col, symbol
        )
    return r2_series


class PerformanceMonitor:
    def __init__(self):
        self.batch_performances = []
        self.r2_records = []
        self.folder = "model_results"

    def record_performance(self, test, pred):
        pred = pred.rename({"responder_6": "responder_6_pred"})
        merged_df = test.with_columns(pred["responder_6_pred"])
        symbol_r2 = self.get_custom_id_r2(
            merged_df["responder_6"].to_numpy(),
            merged_df["responder_6_pred"].to_numpy(),
        )
        merged_df = merged_df.with_columns(pl.Series("symbol_r2", symbol_r2))

        r2_score = self.get_custom_r2(
            merged_df["responder_6"].to_numpy(),
            merged_df["responder_6_pred"].to_numpy(),
            merged_df["weight"].to_numpy() if "weight" in merged_df.columns else None,
        )
        print(f"Custom R² score: {r2_score}")

        date_id = test["date_id"].unique()[0]
        time_id = test["time_id"].unique()[0]
        self.r2_records.append({"date_id": date_id, "time_id": time_id, "r2": r2_score})
        self.batch_performances.append(merged_df)

    def save_results(
        self, performance_path="performance_tracking.parquet", r2_path="r2.parquet"
    ):
        # Concatenate all batches into a single DataFrame
        performance_tracking_df = pl.concat(self.batch_performances)
        performance_tracking_df.write_parquet(f"{self.folder}/" + performance_path)

        # Create a DataFrame for R² records and write to a Parquet file
        r2_df = pl.DataFrame(self.r2_records)
        r2_df.write_parquet(f"{self.folder}/" + r2_path)

    @staticmethod
    def get_custom_r2(y_true, y_pred, weights=None):
        if weights is None:
            weights = np.ones_like(y_true)

        ss_res = np.sum(weights * (y_true - y_pred) ** 2)
        ss_tot = np.sum(weights * (y_true) ** 2)
        return 1 - ss_res / ss_tot

    @staticmethod
    def get_custom_id_r2(y_true, y_pred, weights=None):
        if weights is None:
            weights = np.ones_like(y_true)

        ss_res = (y_true - y_pred) ** 2
        ss_tot = (y_true) ** 2
        return 1 - ss_res / ss_tot
