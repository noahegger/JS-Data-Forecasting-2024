from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge


def get_time_weights(n, halflife=0.35):
    decay_factor = 0.5 ** (1 / (halflife * n))
    weights = decay_factor ** np.arange(n)
    weights /= weights.sum()
    return weights


def time_weighted_mean(vals, n, halflife=0.35):
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


class LinearCalculator(ABC):
    @abstractmethod
    def calculate(self, df):
        raise NotImplementedError("Subclasses should implement this!")


class LinearRegressionCalculator(LinearRegression):
    def __init__(self):
        super().__init__(fit_intercept=False)


class LassoCalculator(LinearCalculator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def calculate(self, df):
        model = Lasso(alpha=self.alpha)
        X = df.drop(columns=["target"])
        y = df["target"]
        model.fit(X, y)
        df["lasso"] = model.predict(X)
        return df


class RidgeCalculator(LinearCalculator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def calculate(self, df):
        model = Ridge(alpha=self.alpha)
        X = df.drop(columns=["target"])
        y = df["target"]
        model.fit(X, y)
        df["ridge"] = model.predict(X)
        return df
