from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import utils as utils


class Calculator(ABC):
    @abstractmethod
    def calculate(self, df):
        raise NotImplementedError("Subclasses should implement this!")


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

    def calculate(self, df: pd.DataFrame, tdate: int, feature_column: str) -> float:
        lookback_dates = range(tdate - self.lookback, tdate)
        mean_values = []

        for date in lookback_dates:
            daily_mean = df[df["date_id"] == date][feature_column].mean()
            mean_values.append(daily_mean)

        nan_count = sum(pd.isna(mean_values))
        if nan_count > self.max_nans:
            return 0

        if self.replace:
            mean_values = [0 if pd.isna(m) else m for m in mean_values]

        res = utils.time_weighted_mean(mean_values, self.lookback, self.halflife)
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


class OnlineMovingAverageCalculator(Calculator):
    def __init__(self, window):
        self.min_periods = window
        self.window = window

    def calculate(
        self, df: pd.DataFrame, tdate: int, feature_column: str
    ) -> pd.DataFrame:
        df = df[df["date_id"] == tdate]
        df[f"online_moving_average_{self.window}"] = (
            df[feature_column]
            .rolling(window=self.window, min_periods=self.min_periods)
            .mean()
        )
        return df[f"online_moving_average_{self.window}"].iloc[-1]
