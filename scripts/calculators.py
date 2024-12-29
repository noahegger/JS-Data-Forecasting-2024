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


class LinearCalculator(ABC):
    @abstractmethod
    def calculate(self, df):
        raise NotImplementedError("Subclasses should implement this!")


class LinearRegressionCalculator(LinearCalculator):
    def calculate(self, df):
        model = LinearRegression()
        X = df.drop(columns=["target"])
        y = df["target"]
        model.fit(X, y)
        df["linear_regression"] = model.predict(X)
        return df


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
