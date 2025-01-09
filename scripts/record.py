from collections import deque
from typing import Dict, List, Union

import pandas as pd
import polars as pl


class SymbolRecord:
    def __init__(self, tdate: int, data: pl.DataFrame):
        self.tdate = tdate
        self.data = data

    def get_feature_series(self, feature_column: str):
        return self.data[feature_column]


class Cache:
    def __init__(self, maxlen: int, freq: int = 1):
        self.cache: Dict[int, deque] = {}
        self.maxlen = maxlen
        self.freq = freq

    def initialize(
        self, data: pl.DataFrame, feature_cols: List[str], lagged: bool = False
    ):
        for (symbol_id,), symbol_data in data.group_by(
            "symbol_id", maintain_order=True
        ):
            if symbol_id not in self.cache:
                self.cache[symbol_id] = deque(maxlen=self.maxlen)  # type: ignore

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
                self.cache[symbol_id].append(SymbolRecord(date_id, batch_data))  # type: ignore

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
            if symbol_id in self.cache and any(
                record.tdate == date_id for record in self.cache[symbol_id]
            ):
                return  # Do nothing if tdate is already in the cache
            else:
                batch_data = batch_data.filter(pl.col("time_id") % self.freq == 0)
                self.cache[symbol_id].append(SymbolRecord(date_id, batch_data))

        else:
            assert batch_data["time_id"].unique().shape[0] == 1

            if batch_data["time_id"].unique()[0] % self.freq != 0:
                return

            if self.cache[symbol_id] and self.cache[symbol_id][-1].tdate == date_id:
                self.cache[symbol_id][-1].data = pl.concat(
                    [self.cache[symbol_id][-1].data, batch_data]
                )
            else:
                self.cache[symbol_id].append(SymbolRecord(date_id, batch_data))
