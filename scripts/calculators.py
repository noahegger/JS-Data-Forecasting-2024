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
    def __init__(self, alpha=1.0, lb: int = 1):
        self.alpha = alpha
        self.lb = lb
        self.models = {}
        self.median_calculator = MedianCalculator()
        self.missing = {}
        self.name = f"Lasso_{alpha}"

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
                    model = Lasso(alpha=self.alpha)
                    # Join X,y and filter top 5% outliers by absolute value
                    # Need corresponding y to also be filtered ... tricky
                    model.fit(X, y)
                    self.models[symbol_id] = model

    def predict(self, symbol_id, X):
        if symbol_id in self.models:
            return self.models[symbol_id].predict(X)
        else:
            # print(f"No model found for symbol_id {symbol_id}")
            return np.array([0], dtype=np.float32)

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

            # Predict and clip estimates
            pred = self.predict(symbol_id, X)
            pred = np.clip(pred, -5, 5)

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

    def calculate_medians(self, data: pl.DataFrame, feature_cols: list):
        for col in feature_cols:
            median_value = data[col].median()
            self.medians[col] = median_value if median_value else np.nan

    def impute(self, data: np.ndarray, feature_cols: list):
        for i, col in enumerate(feature_cols):
            if col in self.medians and not np.isnan(self.medians[col]):
                data[:, i] = np.where(
                    np.isnan(data[:, i]), self.medians[col], data[:, i]
                )
        return data


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

        res = utils.time_weighted_mean(values, self.lookback, self.halflife)
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
