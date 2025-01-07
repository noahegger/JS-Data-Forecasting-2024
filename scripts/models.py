from abc import ABC, abstractmethod
from collections import deque

import numpy as np

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

    def get_daily_estimate(self, data: np.ndarray, tdate: int, feature_column: str):
        return self.online_feature.calculate(data, tdate, feature_column)

    def get_estimate(self, cache: deque, tdate: int, feature_column: str):
        daily_estimates = [
            self.get_daily_estimate(data, tdate, feature_column) for _, data in cache
        ]

        return self.long_term_feature.calculate(daily_estimates, feature_column)

    def get_estimates(
        self,
        cache_history: dict,
        lag_cache: dict,
        symbol_ids: list[int],
        tdate: int,
        ttime: int,
    ):
        estimates = []
        for id in symbol_ids:
            cache_days = cache_history[id]

            estimates.append(self.get_estimate(cache, tdate, feature_columns))

        # for symbol_id, cache_days in cache_history.items():
        #     estimates[symbol_id] = {}
        #     for feature_column in feature_columns:
        #         estimates[symbol_id][feature_column] = self.get_estimate(
        #             cache, tdate, feature_column
        #         )
        return estimates
