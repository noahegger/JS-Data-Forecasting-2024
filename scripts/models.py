from abc import ABC, abstractmethod

import pandas as pd

from scripts.calculators import (
    Calculator,
    ExpWeightedMeanCalculator,
    OnlineMovingAverageCalculator,
    RevDecayCalculator,
)


class BaseModel(ABC):
    @abstractmethod
    def get_estimates(self, calculator: Calculator, tdate: int, feature_column: str):
        raise NotImplementedError("Subclasses should implement this!")


class EnsembleTimeSeriesV1(BaseModel):
    def __init__(
        self,
        context: str,
        online_feature: OnlineMovingAverageCalculator,
        long_term_feature: ExpWeightedMeanCalculator,
        rev_decay_calculator: RevDecayCalculator,
        st_window: int = 15,
        lt_window: int = 15,
    ):
        self.context = context
        self.online_feature = online_feature
        self.long_term_feature = long_term_feature
        self.rev_decay_calculator = rev_decay_calculator
        self.st_window = st_window
        self.lt_window = lt_window

    def get_estimates(self, priors: pd.DataFrame, tdate: int, feature_column: str):
        res = self.online_feature.calculate(priors, tdate, feature_column)
        return res
