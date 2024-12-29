from abc import ABC, abstractmethod

import calculators as calculators
import pandas as pd


class FeatureCalculator(ABC):
    @abstractmethod
    def calculate(self, df):
        raise NotImplementedError("Subclasses should implement this!")


class MovingAverageCalculator(FeatureCalculator):
    def __init__(self, window):
        self.window = window

    def calculate(self, df):
        df[f"moving_average_{self.window}"] = (
            df["responder_6"].rolling(window=self.window).mean()
        )
        return df


class ExpWeightedMeanCalculator(FeatureCalculator):
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

        res = calculators.time_weighted_mean(mean_values, self.lookback, self.halflife)
        return res


class FeatureFactory:
    def __init__(self, calculators):
        self.calculators = calculators

    def create_features(self, df):
        for calculator in self.calculators:
            df = calculator.calculate(df)
        return df
