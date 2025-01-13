from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from calculators import (
    Calculator,
    ExpWeightedMeanCalculator,
    LassoCalculator,
    MeanCalculator,
    RevDecayCalculator,
)
from record import CorrelationCache, SymbolRecord
from sklearn.linear_model import Lasso


class BaseModel(ABC):
    @abstractmethod
    def get_estimates(self, cache_history: dict, tdate: int, feature_columns: list):
        raise NotImplementedError("Subclasses should implement this!")


class EnsembleTimeSeriesV1(BaseModel):
    def __init__(
        self,
        online_feature: MeanCalculator,
        long_term_feature: ExpWeightedMeanCalculator,
        rev_decay_calculator: RevDecayCalculator,
        st_window: int = 15,
        lt_window: int = 15,
    ):
        self.name = "BaseModel"
        self.online_feature = online_feature
        self.long_term_feature = long_term_feature
        self.rev_decay_calculator = rev_decay_calculator
        self.st_window = st_window
        self.lt_window = lt_window
        self.name = f"Base{self.lt_window}d_ExpMean"

    def get_daily_data(self, daily_record: SymbolRecord, feature_column: str):
        return daily_record.data[feature_column]
        # return self.online_feature.calculate(daily_record.data, feature_column)

    def get_estimate(
        self,
        symbol_history: deque[SymbolRecord],
        symbol_lags: deque[SymbolRecord],
        tdate: int,
    ):
        daily_estimates = [
            self.get_daily_data(daily_record, "responder_6_lag_1").to_list()
            for daily_record in symbol_lags
        ][: self.lt_window]
        daily_estimates = [item for sublist in daily_estimates for item in sublist]

        return self.long_term_feature.calculate(daily_estimates)

    def get_estimates(
        self,
        cache_history: dict,
        lag_cache: dict,
        symbol_ids: list[int],
        tdate: int,
        ttime: int,
    ):
        estimates = []
        for symbol in symbol_ids:
            try:
                symbol_history = cache_history[symbol]
                symbol_lags = lag_cache[symbol]
                estimates.append(self.get_estimate(symbol_history, symbol_lags, tdate))
            except KeyError:
                print(
                    f"Symbol: {symbol} not found in cache for tdate, ttime: {tdate, ttime}. Filling with 0"
                )
                estimates.append(0)

        return estimates


class LinearRankedCorrelation(BaseModel):
    def __init__(
        self,
        long_term_feature: ExpWeightedMeanCalculator,
        fitting_model: Lasso,
        max_terms: int = 15,
        lt_window: int = 15,
        st_window: int = 5,
        smoothing_period: int = 20,
    ):
        self.name = "RankedCorrelation"
        self.max_terms = max_terms
        self.fitting_model = fitting_model
        self.long_term_feature = long_term_feature
        self.lt_window = lt_window
        self.st_window = st_window
        self.smoothing_period = smoothing_period
        self.initial_estimate_record = {}

    def get_daily_data(self, daily_record: SymbolRecord, feature_column: str):
        return np.nanmedian(
            daily_record.data[feature_column][: self.smoothing_period]
        )  # return median of first 20 periods for the day
        # return self.online_feature.calculate(daily_record.data, feature_column)

    def get_estimate(
        self,
        symbol_lags: deque[SymbolRecord],
    ):
        daily_estimates = [
            self.get_daily_data(daily_record, "responder_6_lag_1")
            for daily_record in symbol_lags
        ][: self.lt_window]
        # daily_estimates = [item for sublist in daily_estimates for item in sublist]

        return self.long_term_feature.calculate(
            daily_estimates
        )  # exponentially weight the last lt_window medians

    def get_estimates(
        self,
        symbol_ids: list[int],
        cache_history: dict,
        lag_cache: dict,
        corr_cache: CorrelationCache,
        tdate: int,
        ttime: int,
    ):
        estimates = []
        for symbol in symbol_ids:
            if ttime < self.smoothing_period:
                if (symbol, tdate) in self.initial_estimate_record:
                    estimates.append(self.initial_estimate_record[(symbol, tdate)])
                else:
                    try:
                        symbol_history = cache_history[symbol]
                        symbol_lags = lag_cache[symbol]
                        estimate = self.get_estimate(symbol_lags)
                        self.initial_estimate_record[(symbol, tdate)] = estimate
                        estimates.append(estimate)
                    except KeyError:
                        print(
                            f"Symbol: {symbol} not found in cache for tdate, ttime: {tdate, ttime}. Filling with 0"
                        )
                        estimates.append(0)
            else:
                try:
                    symbol_history = cache_history[symbol]
                    lag_history = lag_cache[symbol]
                    day_data = symbol_history[-1].data
                    available_features = [
                        col
                        for col in day_data.columns
                        if "rolling_sign" in col
                        and not day_data[col].is_null().tail(1).item()
                    ]

                    # Get top features
                    top_features = corr_cache.get_top_features(
                        symbol,
                        available_features,
                        top_n=self.max_terms,
                    )

                    # Prepare data for Lasso
                    X = []
                    y = []
                    for record, lag_record in zip(
                        list(symbol_history)[-self.st_window - 1 : -1],
                        list(lag_history)[-self.st_window :],
                    ):
                        X.append(record.data[top_features].to_numpy())
                        y.append(lag_record.data["responder_6_lag_1"].to_numpy())

                    X = np.vstack(X)
                    y = np.concatenate(y)

                    # Drop NaNs
                    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                    X = X[mask]
                    y = y[mask]

                    # Fit Lasso
                    from sklearn.linear_model import LassoCV

                    # model = LassoCV(cv=5).fit(X, y)
                    model = Lasso(
                        alpha=0.1, fit_intercept=False
                    )  # CV(cv=5)  # self.fitting_model
                    model.fit(X, y)

                    # Predict
                    curr_x = day_data[top_features].tail(1).to_numpy()
                    estimates.append(model.predict(curr_x)[0])
                except KeyError:
                    print(
                        f"Symbol: {symbol} not found in cache for tdate, ttime: {tdate, ttime}. Filling with 0"
                    )
                    estimates.append(0)

        return estimates
