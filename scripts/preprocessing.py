import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

DATA_DIR = Path("/Users/noahegger/git/JS-Data-Forecasting-2024")
N_PARTITION = len(os.listdir(DATA_DIR / "train.parquet"))
train_parquets = [
    f"{DATA_DIR}/train.parquet/partition_id={i}/part-0.parquet"
    for i in range(N_PARTITION)
]
test_parquet = f"{DATA_DIR}/lags.parquet/date_id=0"


class Preprocessor:
    def __init__(
        self,
        symbol_id: Optional[int] = None,
        responder: int = 6,
        partition_ids: Optional[list[int]] = None,
        feature_set: Optional[list] = None,
        sample_frequency: int = 15,
        exclude_set: list = [
            "feature_00",
            "feature_01",
            "feature_02",
            "feature_03",
            "feature_04",
            "feature_21",
            "feature_26",
            "feature_27",
            "feature_31",
        ],
    ):
        self.symbol_id = symbol_id
        self.responder = responder
        self.partition_ids = partition_ids
        self.feature_set = feature_set
        self.sample_frequency = sample_frequency
        self.exclude_set = exclude_set

    def filter_symbol(self, df: pd.DataFrame, symbol_id: int):
        return df[df["symbol_id"] == symbol_id]

    def resample(self, df: pd.DataFrame, sample_frequency: int):
        df.set_index("time_index", inplace=True)
        df = df.resample(f"{sample_frequency}min").first().reset_index()
        return df

    def create_time_index(self, df: pd.DataFrame):
        # Convert to numpy.int32 to prevent overflow
        df["date_id"] = df["date_id"].astype("int32")
        df["time_id"] = df["time_id"].astype("int32")

        # Create a custom date index
        df["day"] = df["date_id"]
        df["hour"] = df["time_id"] // 60 + 4
        df["minute"] = df["time_id"] % 60
        df["time_index"] = (
            pd.to_datetime(df["day"], unit="D")
            + pd.to_timedelta(df["hour"], unit="h")
            + pd.to_timedelta(df["minute"], unit="m")
        )

    def read_partition(self, read_all=False) -> Dict[str, pd.DataFrame]:
        if self.partition_ids:
            if read_all:
                main_df = pd.concat(
                    [pd.read_parquet(parquet) for parquet in train_parquets]
                )
            else:
                dfs = []
                for partition_id in self.partition_ids:
                    dfs.append(pd.read_parquet(train_parquets[partition_id]))
                main_df = pd.concat(dfs, ignore_index=True)
        else:
            main_df = pd.read_parquet(test_parquet)

        symbol_dfs = {}
        for symbol_id, group in main_df.groupby("symbol_id"):
            self.create_time_index(group)
            group = self.resample(group, self.sample_frequency)
            if self.exclude_set:
                group.drop(columns=self.exclude_set, inplace=True)
            symbol_dfs[f"symbol_{symbol_id:02}"] = group

        return symbol_dfs

    def windsorize(
        self,
        df: pd.DataFrame,
        lower_percentile: float = 0.02,
        upper_percentile: float = 0.98,
    ):
        lower_bound = df.quantile(lower_percentile)
        upper_bound = df.quantile(upper_percentile)
        df.clip(lower=lower_bound, upper=upper_bound, axis=1, inplace=True)

    def map_symbols_to_dfs(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        symbol_dfs = {}
        for symbol_id in df["symbol_id"].unique():
            symbol_df = self.filter_symbol(df, symbol_id)
            symbol_dfs[f"symbol_{symbol_id:02}"] = symbol_df
        return symbol_dfs
