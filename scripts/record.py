from collections import defaultdict, deque
from itertools import combinations
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr


class SymbolRecord:
    def __init__(
        self,
        tdate: int,
        data: pl.DataFrame,
        freq: int = 1,
        cross_term: Optional[str] = None,
    ):
        self.tdate = tdate
        self.data = data
        self.freq = freq
        self.cross_term = cross_term

    def get_feature_series(self, feature_column: str):
        return self.data[feature_column]  # .shift(-int(20/self.freq))

    def add_smoothed_response(self, period: int = 20):
        mean_response = self.data["responder_6"].rolling_mean(period)
        self.data = self.data.with_columns(mean_response.alias("responder_6_smoothed"))

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
                rolling_sign_mean = rolling_sign.rolling_mean(period)

                if self.cross_term is None or self.cross_term == "std":
                    rolling_std = self.data[col].rolling_std(period)
                    prod_std = rolling_sign_mean * rolling_std
                    self.data = self.data.with_columns(
                        prod_std.alias(f"{col}_rolling_sign_std")
                    )
                    # self.data = self.data.with_columns(
                    #     rolling_std.alias(f"{col}_rolling_sign_std")
                    # )

                if self.cross_term is None or self.cross_term == "mean":
                    rolling_mean = self.data[col].rolling_mean(period)
                    prod_mean = rolling_sign_mean * rolling_mean
                    self.data = self.data.with_columns(
                        prod_mean.alias(f"{col}_rolling_sign_mean")
                    )
                    # self.data = self.data.with_columns(
                    #     rolling_mean.alias(f"{col}_rolling_sign_mean")
                    # )

                # if self.cross_term is None or self.cross_term == "med":
                #     rolling_med = self.data[col].rolling_median(period)
                #     # prod_mean = rolling_sign_mean * rolling_mean
                #     # self.data = self.data.with_columns(
                #     #     prod_mean.alias(f"{col}_rolling_sign_mean")
                #     # )
                #     self.data = self.data.with_columns(
                #         rolling_med.alias(f"{col}_rolling_sign_med")
                #     )

                if self.cross_term is None or self.cross_term == "self":
                    prod_self = rolling_sign_mean * self.data[col]
                    self.data = self.data.with_columns(
                        prod_self.alias(f"{col}_rolling_sign_self")
                    )

                # if self.cross_term is None:
                self.data = self.data.with_columns(
                    rolling_sign_mean.alias(f"{col}_rolling_sign")
                )
        # feature_columns = [
        #     col
        #     for col in self.data.columns
        #     if "feature" in col and "_rolling_sign" not in col
        # ]
        # feature_pairs = combinations(feature_columns, 2)

        # for col1, col2 in feature_pairs:
        #     new_feature_name = f"{col1}_rolling_sign_{col2}"
        #     if new_feature_name not in self.data.columns:
        #         prod = self.data[col1] * self.data[col2]
        #         self.data = self.data.with_columns(prod.alias(new_feature_name))


class SymbolCorrRecord:
    def __init__(self, tdate: int, corr):
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
                if lagged:
                    record.add_smoothed_response(self.smoothing_period)
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
            record = SymbolRecord(date_id, batch_data)
            record.add_smoothed_response(self.smoothing_period)
            self.cache[symbol_id].append(record)

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
                    # lag_data = lag_record.data["responder_6_lag_1"].to_numpy()
                    lag_data = lag_record.data["responder_6_smoothed"].to_numpy()

                    # Create masked arrays to ignore NaNs
                    feature_data_masked = np.ma.masked_invalid(feature_data)
                    lag_data_masked = np.ma.masked_invalid(lag_data)

                    valid_mask = ~feature_data_masked.mask & ~lag_data_masked.mask
                    feature_data_aligned = feature_data_masked[valid_mask]
                    lag_data_aligned = lag_data_masked[valid_mask]
                    # Calculate Spearman correlation using masked arrays
                    # Require at least 500 data points of the day to calculate correlation
                    ## For example, if only 10 data points are used to calculate a corr, not useful
                    if len(feature_data_aligned) > 500 and len(lag_data_aligned) > 500:
                        corr, _ = spearmanr(feature_data_aligned, lag_data_aligned)

                    else:
                        corr = np.nan

                    self.cache[symbol_id][feature].append(
                        SymbolCorrRecord(date_id, corr)  # type: ignore
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
            # lag_data = lag_record.data["responder_6_lag_1"].to_numpy()
            lag_data = lag_record.data["responder_6_smoothed"].to_numpy()

            # Create masked arrays to ignore NaNs
            feature_data_masked = np.ma.masked_invalid(feature_data)
            lag_data_masked = np.ma.masked_invalid(lag_data)
            valid_mask = ~feature_data_masked.mask & ~lag_data_masked.mask
            feature_data_aligned = feature_data_masked[valid_mask]
            lag_data_aligned = lag_data_masked[valid_mask]

            # Calculate Pearson correlation using masked arrays
            if len(feature_data_aligned) > 1 and len(lag_data_aligned) > 1:
                corr, _ = spearmanr(feature_data_aligned, lag_data_aligned)

            else:
                corr = np.nan

            self.cache[symbol_id][feature].append(SymbolCorrRecord(date_id, corr))  # type: ignore

    def get_top_features(
        self,
        symbol_id: int,
        available_features: List[str],
        top_n: int = 10,
        threshold: float = 0.15,
        min_records: int = 7,
    ):  # Probably not working correctly
        def has_min_consecutive_same_sign(correlations, min_records):
            count = 0
            sign = None
            for corr in correlations:
                if np.isnan(corr):
                    count = 0
                    sign = None
                else:
                    current_sign = np.sign(corr)
                    if sign is None:
                        sign = current_sign
                        count = 1
                    elif current_sign == sign:
                        count += 1
                        if count >= min_records:
                            return True
                    else:
                        count = 1
                        sign = current_sign
            return False

        feature_correlations = {}
        for feature, records in self.cache[symbol_id].items():
            if feature in available_features:
                # Collect all correlation values for the feature
                correlations = [record.corr for record in records]
                # Ensure there are at least min_records consecutive correlations with the same sign
                if has_min_consecutive_same_sign(correlations, min_records):
                    # Calculate the mean of the absolute values of the correlations
                    abs_correlations = np.abs(correlations)
                    mean_abs_corr = np.mean(abs_correlations)
                    # Check if mean_abs_corr is above the threshold
                    if mean_abs_corr > threshold:
                        feature_correlations[feature] = mean_abs_corr
        feature_correlations = {
            k: v for k, v in feature_correlations.items() if not np.isnan(v)
        }

        sorted_features = sorted(
            feature_correlations, key=lambda x: feature_correlations[x], reverse=True
        )

        return sorted_features[:top_n], [
            feature_correlations[feature] for feature in sorted_features[:top_n]
        ]
