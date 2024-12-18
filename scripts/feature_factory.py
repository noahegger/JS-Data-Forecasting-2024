from abc import ABC, abstractmethod

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


class FeatureFactory:
    def __init__(self, calculators):
        self.calculators = calculators

    def create_features(self, df):
        for calculator in self.calculators:
            df = calculator.calculate(df)
        return df
