from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import polars as pl
import utils as utils
from record import SymbolRecord
from sklearn.linear_model import Lasso, LinearRegression, Ridge


class Calculator(ABC):
    @abstractmethod
    def calculate(self, df):
        raise NotImplementedError("Subclasses should implement this!")


class LinearCalculator(ABC):
    @abstractmethod
    def calculate(self, df):
        raise NotImplementedError("Subclasses should implement this!")


class LinearRegressionCalculator(LinearRegression):
    def __init__(self):
        super().__init__(fit_intercept=False)


class LassoCalculator:
    def __init__(
        self,
        alpha=1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        lb=1,
        truncate_calculator=None,
        intercept=False,
    ):
        self.alpha = alpha
        self.intercept = intercept
        self.max_iter = max_iter
        self.tol = tol
        self.lb = lb
        self.models = {}
        self.median_calculator = MedianCalculator()
        self.feature_calculator = ZscoreCalculator()
        self.truncate_calculator = truncate_calculator
        self.missing = {}
        self.name = f"Lasso_{alpha}"
        self.valid_features = {}

    def fit(self, symbol_ids, test_data, cache_history, lag_cache, feature_cols):
        for symbol_id in symbol_ids:
            if symbol_id in lag_cache.cache and len(lag_cache.cache[symbol_id]) > 0:
                # Get current features and find intersection with availale
                symbol_data = test_data.filter(pl.col("symbol_id") == symbol_id)
                present_features = [
                    col
                    for col in symbol_data.columns
                    if "feature" in col and not symbol_data[col].is_null().any()
                ]

                # Features we were given and also willing to use
                intersected_features = list(set(feature_cols) & set(present_features))
                # Determine the lookback period based on the minimum length of lag_cache and cache_history
                lookback = min(
                    self.lb,
                    len(lag_cache.cache[symbol_id]),
                    len(cache_history.cache[symbol_id]) + 1,
                )

                # Get the most recent elements of the lag_cache for y values
                y_records = list(lag_cache.cache[symbol_id])[-lookback:]
                y = np.concatenate(
                    [
                        record.get_feature_series("responder_6_lag_1")
                        .to_numpy()
                        .ravel()
                        for record in y_records
                    ]
                )

                # Get the corresponding elements of the cache_history for X values
                if len(cache_history.cache[symbol_id]) > lookback:
                    x_records = list(cache_history.cache[symbol_id])[-lookback - 1 : -1]
                    X = np.concatenate(
                        [
                            record.data.select(intersected_features).to_numpy()
                            for record in x_records
                        ],
                        axis=0,
                    )

                    # Calculate medians and impute NaNs
                    self.median_calculator.calculate_medians(X, intersected_features)
                    X = self.median_calculator.impute(X, intersected_features)

                    # Apply feature transformations : USE Z-SCORES
                    if self.truncate_calculator:
                        X, y = self.truncate_calculator.truncate(X, y)

                    # Apply feature transformations: Transform to z-scores
                    X = self.feature_calculator.transform(X, intersected_features)

                    # Identify and store features that are entirely NaN in past data
                    self.missing[symbol_id] = []
                    for i, col in enumerate(intersected_features):
                        if np.isnan(self.median_calculator.medians[col]):
                            self.missing[symbol_id].append(col)

                    # Remove entirely NaN features from fitting data and store info
                    valid_indices = [
                        i
                        for i, col in enumerate(intersected_features)
                        if col not in self.missing[symbol_id]
                    ]
                    valid_features = [
                        col
                        for col in intersected_features
                        if col not in self.missing[symbol_id]
                    ]
                    self.valid_features[symbol_id] = valid_features
                    X = X[:, valid_indices]

                    # Fit the model for the current symbol_id
                    model = Lasso(
                        alpha=self.alpha,
                        fit_intercept=self.intercept,
                        max_iter=self.max_iter,
                        tol=self.tol,
                    )
                    model.fit(X, y)
                    print("Model Coefficients:", model.coef_)
                    self.models[symbol_id] = model

    def predict(self, symbol_id, X):
        pred = self.models[symbol_id].predict(X)
        pred = np.clip(pred, -5, 5)
        return pred

    def get_estimates(self, symbol_ids, test_data, feature_cols):
        estimates = []
        for symbol_id in symbol_ids:
            symbol_data = test_data.filter(pl.col("symbol_id") == symbol_id)

            if symbol_id in self.models:
                X = (
                    symbol_data.select(self.valid_features[symbol_id])
                    .to_numpy()
                    .astype(np.float32)
                )
                X = self.feature_calculator.transform_single_record(
                    X, self.valid_features[symbol_id]
                )

                pred = self.predict(symbol_id, X)
            else:
                pred = np.array([0], dtype=np.float32)
            estimates.append(pred)
        return np.concatenate(estimates)


class RidgeCalculator:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.models = {}
        self.median_calculator = MedianCalculator()
        self.missing = {}
        self.name = f"Ridge_{alpha}"

    def fit(self, symbol_ids, cache_history, lag_cache, feature_cols):
        for symbol_id in symbol_ids:
            if symbol_id in lag_cache.cache and len(lag_cache.cache[symbol_id]) > 0:
                # Get the most recent element of the lag_cache for y values
                y_record = lag_cache.cache[symbol_id][-1]
                y = y_record.get_feature_series("responder_6_lag_1").to_numpy().ravel()

                # Get the second most recent element of the cache_history for X values
                if len(cache_history.cache[symbol_id]) > 1:
                    x_record = cache_history.cache[symbol_id][-2]
                    X = x_record.data.select(feature_cols).to_numpy()

                    # Calculate medians and impute NaNs
                    self.median_calculator.calculate_medians(
                        x_record.data, feature_cols
                    )
                    X = self.median_calculator.impute(X, feature_cols)

                    # Identify and store features that are entirely NaN
                    self.missing[symbol_id] = []
                    for i, col in enumerate(feature_cols):
                        if np.isnan(self.median_calculator.medians[col]):
                            self.missing[symbol_id].append(col)

                    # Remove entirely NaN features from X and feature_cols
                    valid_features = [
                        col
                        for col in feature_cols
                        if col not in self.missing[symbol_id]
                    ]
                    valid_indices = [
                        i
                        for i, col in enumerate(feature_cols)
                        if col not in self.missing[symbol_id]
                    ]
                    X = X[:, valid_indices]

                    # Fit the model for the current symbol_id
                    model = Ridge(alpha=self.alpha)
                    model.fit(X, y)
                    self.models[symbol_id] = model

    def predict(self, symbol_id, X):
        if symbol_id in self.models:
            return self.models[symbol_id].predict(X)
        else:
            print(f"No model found for symbol_id {symbol_id}")
            return 0

    def get_estimates(self, symbol_ids, test_data, feature_cols):
        estimates = []
        for symbol_id in symbol_ids:
            symbol_data = test_data.filter(pl.col("symbol_id") == symbol_id)
            valid_features = [
                col
                for col in feature_cols
                if col not in self.missing.get(symbol_id, [])
            ]
            X = symbol_data.select(valid_features).to_numpy()

            # Impute NaNs with medians
            X = self.median_calculator.impute(X, valid_features)

            estimates.append(np.clip(self.predict(symbol_id, X), -5, 5))
        return np.concatenate(estimates)


class MedianCalculator:
    def __init__(self):
        self.medians = {}

    def calculate_medians(self, data: np.ndarray, feature_cols: list):
        for i, col in enumerate(feature_cols):
            median_value = np.nanmedian(data[:, i])
            self.medians[col] = median_value if not np.isnan(median_value) else np.nan

    def impute(self, data: np.ndarray, feature_cols: list):
        for i, col in enumerate(feature_cols):
            if col in self.medians and not np.isnan(self.medians[col]):
                data[:, i] = np.where(
                    np.isnan(data[:, i]), self.medians[col], data[:, i]
                )
        return data


class ZscoreCalculator:
    def __init__(self):
        self.means = {}
        self.stds = {}

    def transform(self, data: np.ndarray, feature_cols: list):
        for i, col in enumerate(feature_cols):
            mean = np.nanmean(data[:, i])
            std = np.nanstd(data[:, i])
            self.means[col] = mean
            self.stds[col] = std
            data[:, i] = (data[:, i] - mean) / std if std != 0 else 0
        return data

    def transform_single_record(self, data: np.ndarray, feature_cols: list):
        z_scores = np.zeros_like(data)
        for i, col in enumerate(feature_cols):
            mean = self.means[col]
            std = self.stds[col]
            z_scores[:, i] = (data[:, i] - mean) / std if std != 0 else 0
        return z_scores


class TruncateCalculator:
    def __init__(self, lower_percentile=2.5, upper_percentile=97.5):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def truncate(self, X, y):
        lower_bound = np.percentile(y, self.lower_percentile, axis=0)
        upper_bound = np.percentile(y, self.upper_percentile, axis=0)
        mask = (y >= lower_bound) & (y <= upper_bound)
        return X[mask], y[mask]


class MovingAverageCalculator(Calculator):
    def __init__(self, window):
        self.window = window

    def calculate(self, df):
        df[f"moving_average_{self.window}"] = (
            df["responder_6"].rolling(window=self.window).mean()
        )
        return df


class ExpWeightedMeanCalculator(Calculator):
    def __init__(self, halflife=0.35, lookback=15, max_nans=5, replace=True):
        self.halflife = halflife
        self.lookback = lookback
        self.max_nans = max_nans
        self.replace = replace

    def calculate(self, values: list) -> float:
        if len(values) < self.lookback:
            return sum(values) / len(values) if values else 0

        nan_count = sum(pd.isna(values))
        if nan_count > self.max_nans:
            return 0

        if self.replace:
            values = [0 if pd.isna(m) else m for m in values]

        res = utils.time_weighted_mean(values, self.halflife)
        return res


class RevDecayCalculator(Calculator):
    def __init__(self, lookback=15, max_fails=5, replace=False):
        self.lookback = lookback
        self.max_fails = max_fails
        self.replace = replace

    def calculate(
        self, df: pd.DataFrame, pred_val: float, tdate: int, feature_column: str
    ) -> float:
        lookback_dates = range(tdate - self.lookback, tdate)
        values = []
        failures = 0
        for date in lookback_dates:
            daily_values = df[df["date_id"] == date][feature_column].tolist()
            if len(daily_values) == 0:
                falilures += 1
            else:
                values.extend(daily_values)

        if failures > self.max_fails:
            return pred_val

        abs_values = np.abs(values)
        percentiles = np.percentile(abs_values, [50, 95])

        decayed_values = [self.blend_decay(v, percentiles) for v in values]
        return decayed_values[-1]

    def blend_decay(self, value, percentiles):
        abs_value = abs(value)
        percentile = np.searchsorted(percentiles, abs_value) / len(percentiles)
        alpha = 0.5 + (1 - percentile)
        return value * alpha


class MeanCalculator(Calculator):
    def __init__(self, window: int):
        self.window = window

    def calculate(self, data: pl.DataFrame, feature_column: str):
        # Calculate the moving average over the window
        return data[feature_column].mean()
