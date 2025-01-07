from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from record import SymbolRecord

from scripts.calculators import (
    Calculator,
    ExpWeightedMeanCalculator,
    OnlineMovingAverageCalculator,
    RevDecayCalculator,
)


class BaseModel(ABC):
    @abstractmethod
    def get_estimates(self, cache_history: dict, tdate: int, feature_columns: list):
        raise NotImplementedError("Subclasses should implement this!")


class EnsembleTimeSeriesV1(BaseModel):
    def __init__(
        self,
        online_feature: OnlineMovingAverageCalculator,
        long_term_feature: ExpWeightedMeanCalculator,
        rev_decay_calculator: RevDecayCalculator,
        st_window: int = 15,
        lt_window: int = 15,
    ):
        self.online_feature = online_feature
        self.long_term_feature = long_term_feature
        self.rev_decay_calculator = rev_decay_calculator
        self.st_window = st_window
        self.lt_window = lt_window

    def get_daily_estimate(self, daily_record: SymbolRecord, feature_column: str):
        return self.online_feature.calculate(daily_record.data, feature_column)

    def get_estimate(
        self,
        symbol_history: deque[SymbolRecord],
        symbol_lags: deque[SymbolRecord],
        tdate: int,
    ):
        daily_estimates = [
            self.get_daily_estimate(daily_record, "responder_6_lag_1")
            for daily_record in symbol_lags
        ]

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
