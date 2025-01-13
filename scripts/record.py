from collections import defaultdict, deque
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl


class SymbolRecord:
    def __init__(self, tdate: int, data: pl.DataFrame, freq: int = 1):
        self.tdate = tdate
        self.data = data
        self.freq = freq

    def get_feature_series(self, feature_column: str):
        return self.data[feature_column]  # .shift(-int(20/self.freq))

    def add_derived_features(self, period: int = 20):
        for col in self.data.columns:
            if "feature" in col and "_rolling_sign" not in col:
                rolling_sign = (
                    pl.when(pl.col(col) > 0)
                    .then(1)
                    .when(pl.col(col) < 0)
                    .then(-1)
                    .otherwise(0)
                )
                self.data = self.data.with_columns(
                    rolling_sign.rolling_mean(period).alias(f"{col}_rolling_sign")
                )


class SymbolCorrRecord:
    def __init__(self, tdate: int, corr: float):
        self.tdate = tdate
        self.corr = corr


class Cache:
    def __init__(self, maxlen: int, smoothing_period: int, freq: Optional[int] = None):
        self.cache: Dict[int, deque[SymbolRecord]] = {}
        self.maxlen = maxlen
        self.freq = freq
        self.smoothing_period = smoothing_period

    def initialize(
        self, data: pl.DataFrame, feature_cols: List[str], lagged: bool = False
    ):
        for (symbol_id,), symbol_data in data.group_by(
            "symbol_id", maintain_order=True
        ):
            if symbol_id not in self.cache:
                self.cache[symbol_id] = deque(maxlen=self.maxlen)  # type: ignore

            if self.freq:
                symbol_data = symbol_data.filter(pl.col("time_id") % self.freq == 0)

            for (date_id,), date_data in symbol_data.group_by(
                "date_id", maintain_order=True
            ):

                batch_data = date_data.select(feature_cols)
                if lagged:
                    batch_data = batch_data.rename(
                        {f"responder_{x}": f"responder_{x}_lag_1" for x in range(9)}
                    )
                    date_id += 1  # type: ignore
                    batch_data = batch_data.with_columns(
                        (pl.col("date_id") + 1).alias("date_id")
                    )
                record = SymbolRecord(date_id, batch_data)  # type: ignore
                if not lagged:
                    record.add_derived_features(
                        self.smoothing_period
                    )  # cache history gets the new features
                self.cache[symbol_id].append(record)  # type: ignore

    def update(
        self,
        symbol_id: int,
        date_id: int,
        batch_data: pl.DataFrame,
        is_lag_cache: bool = False,
    ):
        if symbol_id not in self.cache:
            self.cache[symbol_id] = deque(maxlen=self.maxlen)

        if is_lag_cache:
            # Check if the tdate is already in the cache
            # if symbol_id in self.cache and any(
            #     record.tdate == date_id for record in self.cache[symbol_id]
            # ):
            #     return  # Do nothing if tdate is already in the cache
            # else:
            batch_data = batch_data.filter(pl.col("time_id") % self.freq == 0)
            self.cache[symbol_id].append(SymbolRecord(date_id, batch_data))

        else:
            assert batch_data["time_id"].unique().shape[0] == 1

            if self.freq:
                if batch_data["time_id"].unique()[0] % self.freq != 0:
                    return

            if self.cache[symbol_id] and self.cache[symbol_id][-1].tdate == date_id:
                # combined_data = pl.concat(
                #     [self.cache[symbol_id][-1].data, batch_data]
                # )
                # updated_data =
                existing_columns = self.cache[symbol_id][-1].data.columns
                for col in existing_columns:
                    if col not in batch_data.columns:
                        batch_data = batch_data.with_columns(pl.lit(None).alias(col))

                # Append the new row to the existing DataFrame
                self.cache[symbol_id][-1].data = pl.concat(
                    [self.cache[symbol_id][-1].data, batch_data], how="vertical"
                )

                # Recalculate derived features for the updated DataFrame
                if batch_data["time_id"].unique()[0] >= self.smoothing_period:
                    self.cache[symbol_id][-1].add_derived_features(
                        self.smoothing_period
                    )

            else:
                record = SymbolRecord(date_id, batch_data)
                record.add_derived_features(self.smoothing_period)
                self.cache[symbol_id].append(record)


class CorrelationCache:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.cache: Dict[int, Dict[str, deque[SymbolCorrRecord]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.maxlen))
        )
        # defaultdict(
        #     lambda: defaultdict(deque)
        # )

    def initialize(
        self,
        cache_history: Dict[int, deque[SymbolRecord]],
        lag_cache: Dict[int, deque[SymbolRecord]],
    ):
        for symbol_id in cache_history.keys():
            for record, lag_record in zip(
                cache_history[symbol_id],
                lag_cache[symbol_id],  # will need to check that carefully
            ):
                full_feature_list = record.data.columns
                feature_list = [
                    col for col in full_feature_list if "rolling_sign" in col
                ]
                date_id = record.tdate
                for feature in feature_list:
                    feature_data = record.data[feature].to_numpy()
                    lag_data = lag_record.data["responder_6_lag_1"].to_numpy()

                    # Create masked arrays to ignore NaNs
                    feature_data_masked = np.ma.masked_invalid(feature_data)
                    lag_data_masked = np.ma.masked_invalid(lag_data)

                    # Calculate Pearson correlation using masked arrays
                    corr = np.ma.corrcoef(feature_data_masked, lag_data_masked)[0, 1]
                    if np.ma.is_masked(corr):
                        corr = np.nan

                    self.cache[symbol_id][feature].append(
                        SymbolCorrRecord(date_id, corr)
                    )

    def update(
        self,
        symbol_id: int,
        date_id: int,
        cache_history: Dict[int, deque[SymbolRecord]],
        lag_cache: Dict[int, deque[SymbolRecord]],
    ):  # corr cache is only updated when lags are passed in, so dates are offset by 1
        # correlation get from other cache
        record = cache_history[symbol_id][-1]
        lag_record = lag_cache[symbol_id][-1]
        full_feature_list = record.data.columns
        feature_list = [col for col in full_feature_list if "rolling_sign" in col]
        date_id = record.tdate
        for feature in feature_list:
            feature_data = record.data[feature].to_numpy()
            lag_data = lag_record.data["responder_6_lag_1"].to_numpy()

            # Create masked arrays to ignore NaNs
            feature_data_masked = np.ma.masked_invalid(feature_data)
            lag_data_masked = np.ma.masked_invalid(lag_data)

            # Calculate Pearson correlation using masked arrays
            corr = np.ma.corrcoef(feature_data_masked, lag_data_masked)[0, 1]
            if np.ma.is_masked(corr):
                corr = np.nan

            self.cache[symbol_id][feature].append(SymbolCorrRecord(date_id, corr))

    def get_top_features(
        self, symbol_id: int, available_features: List[str], top_n: int = 10
    ):  # Probably not working correctly
        feature_correlations = {}
        for feature, records in self.cache[symbol_id].items():
            if feature in available_features:
                # Collect all correlation values for the feature
                correlations = [record.corr for record in records]
                # Calculate the mean of the absolute values of the correlations
                if all(c > 0 for c in correlations) or all(c < 0 for c in correlations):
                    # Calculate the mean of the absolute values of the correlations
                    abs_correlations = np.abs(correlations)
                    mean_abs_corr = np.mean(abs_correlations)
                    feature_correlations[feature] = mean_abs_corr
        feature_correlations = {
            k: v for k, v in feature_correlations.items() if not np.isnan(v)
        }

        sorted_features = sorted(
            feature_correlations, key=lambda x: feature_correlations[x], reverse=True
        )
        return sorted_features[:top_n]
