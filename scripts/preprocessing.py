import os
from pathlib import Path
from typing import Optional

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
        return (
            df.set_index("time_index")
            .resample(f"{sample_frequency}min")
            .first()
            .reset_index()
        )

    def create_time_index(self, df: pd.DataFrame) -> pd.DataFrame:

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

        return df

    def read_partition(self, read_all=False) -> pd.DataFrame:
        if self.partition_ids:
            if read_all:
                df = pd.concat([pd.read_parquet(parquet) for parquet in train_parquets])
            else:
                dfs = []
                for partition_id in self.partition_ids:
                    dfs.append(pd.read_parquet(train_parquets[partition_id]))
                df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_parquet(test_parquet)

        df = self.create_time_index(df)
        if self.symbol_id:
            df = self.filter_symbol(df, self.symbol_id)
        df = self.resample(df, self.sample_frequency)

        if self.exclude_set:
            df.drop(columns=self.exclude_set, inplace=True)

        return df
