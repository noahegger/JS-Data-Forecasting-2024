import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import IO, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# Define a global variable for the project path
DATA_DIR = Path("/kaggle/input/jane-street-real-time-market-data-forecasting")
LOCAL_DATA_DIR = Path("/Users/noahegger/git/JS-Data-Forecasting-2024")
PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_PATH))

META_COLS = ["date_id", "time_id", "symbol_id", "weight"]
FEATURE_COLS = [f"feature_{x:02}" for x in range(79)]
RESPONDER_COLS = [f"responder_{i}" for i in range(9)]

from scripts.calculators import (
    ExpWeightedMeanCalculator,
    MovingAverageCalculator,
    OnlineMovingAverageCalculator,
    RevDecayCalculator,
)
from scripts.models import BaseModel, EnsembleTimeSeriesV1
from scripts.preprocessing import Preprocessor


class InferenceServer:
    def __init__(
        self,
        model,
        preprocessor,
        dir: Optional[Path] = None,
        test_parquet: Optional[str] = None,
        lag_parquet: Optional[str] = None,
        cache_lb_days: int = 15,
        feature_cols: List[str] = ["responder_6"],
        test: bool = False,
        partition_ids: Optional[List[int]] = None,
        synthetic_days: int = 50,
        num_score_dates: int = 5,
        date_offset: int = 1690,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.dir = dir
        # if test_parquet is None or lag_parquet is None:
        #     import warnings

        #     warnings.warn(
        #         "File paths for test_parquet or lag_parquet are not specified."
        #     )
        self.test_parquet = str(test_parquet) if test_parquet else ""
        self.lag_parquet = str(lag_parquet) if lag_parquet else ""
        self.cache_lb_days = cache_lb_days
        self.feature_cols = feature_cols
        self.cache_history = {}
        self.test_ = None
        self.lags_ = None
        self.time_step_count = 0
        self.test = test
        self.partition_ids = partition_ids
        self.synthetic_days = synthetic_days
        self.num_score_dates = num_score_dates
        self.date_offset = date_offset

        self.ensure_parquet_files()
        self.total_time_steps = self.calculate_total_time_steps()
        self.initialize_pbar()

    def ensure_parquet_files(self):
        if self.test and (
            not self.test_parquet
            or not Path(self.test_parquet).exists()
            or not self.lag_parquet
            or not Path(self.lag_parquet).exists()
        ):
            self.test_parquet, self.lag_parquet = self.generate_synthetic_data()

    def generate_synthetic_data(self):
        # Determine the parquet files to read based on partition_ids
        train_parquets = [
            f"{self.dir}/train.parquet/partition_id={i}/part-0.parquet"
            for i in (self.partition_ids or range(10))
        ]

        # Read and concatenate the parquet files
        train_data = pl.concat([pl.read_parquet(parquet) for parquet in train_parquets])

        # Filter data based on synthetic_days
        # if self.synthetic_days:
        date_ids = sorted(train_data["date_id"].unique())[: self.synthetic_days]
        test_data, lag_data = (
            train_data.filter(pl.col("date_id").is_in(date_ids)),
            train_data.filter(pl.col("date_id").is_in(date_ids)),
        )
        # else:
        #     test_data, lag_data = train_data, train_data

        # add row_id column, offset by the last date_id
        # should be 1690 if using the entire train_parquet

        test_data = test_data.with_columns(
            pl.col("date_id") - self.cache_lb_days
        ).with_row_index(name="row_id", offset=0)

        # num_scored_days = self.synthetic_days - self.cache_lb_days
        # Add "is_scored" column to test_data
        test_data = test_data.with_columns(
            pl.when(pl.col("date_id") <= 0)
            .then(False)
            .otherwise(True)
            .alias("is_scored")
        )

        # Select relevant columns for test_data and lag_data
        test_data = test_data.select(
            ["row_id"] + META_COLS + ["is_scored"] + FEATURE_COLS
        )
        lag_data = lag_data.select(META_COLS + RESPONDER_COLS)

        # Define paths for synthetic data
        test_parquet_path = Path(self.test_parquet)
        lag_parquet_path = Path(self.lag_parquet)

        # Create directories if they don't exist
        test_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        lag_parquet_path.parent.mkdir(parents=True, exist_ok=True)

        # Partition and save test_data
        test_data_partition = test_data.partition_by(
            "date_id", maintain_order=True, as_dict=True
        )
        row_id_offset = (
            test_data.filter(pl.col("date_id") < 0).select("row_id").max().item()
        )

        for key, _df in test_data_partition.items():
            if key[0] >= 0:
                partition_dir = test_parquet_path / f"date_id={key[0]}"
                partition_dir.mkdir(parents=True, exist_ok=True)
                _df = _df.with_columns(pl.col("row_id") - row_id_offset)
                _df.write_parquet(partition_dir / "part-0.parquet")

        # Partition and save lag_data
        lag_data = lag_data.with_columns(pl.col("date_id") - self.cache_lb_days)
        lag_data = lag_data.rename(
            {f"responder_{x}": f"responder_{x}_lag_1" for x in range(9)}
        )
        lag_data_partition = lag_data.partition_by(
            "date_id", maintain_order=True, as_dict=True
        )

        for key, _df in lag_data_partition.items():
            partition_dir = lag_parquet_path / f"date_id={key[0] + 1}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            _df = _df.with_columns(pl.col("date_id") + 1)
            _df.write_parquet(partition_dir / "part-0.parquet")

        return str(test_parquet_path), str(lag_parquet_path)

    def calculate_total_time_steps(self):
        if self.test and self.test_parquet:
            return (
                pl.scan_parquet(self.test_parquet)
                .select((pl.col("date_id") * 10000 + pl.col("time_id")).n_unique())
                .collect()
                .item()
            )
        return 1  # For live submission

    def initialize_pbar(self):
        self.pbar = tqdm(total=self.total_time_steps)

    def generate_data_batches(self, test_data: pl.DataFrame, lag_data: pl.DataFrame):
        date_ids = sorted(test_data["date_id"].unique())
        assert date_ids[0] == 0

        for date_id in date_ids:
            test_batches = test_data.filter(pl.col("date_id") == date_id).group_by(
                "time_id", maintain_order=True
            )
            lags = lag_data.filter(pl.col("date_id") == date_id)

            for (time_id,), test in test_batches:
                test_batch = (test, lags if time_id == 0 else None)
                validation_data = test.select("row_id")  # row_id in gateway
                yield test_batch, validation_data

    def run_inference_server(self):
        self.pbar.refresh()

        if self.test:
            if self.lag_parquet and self.test_parquet:
                lag_data = pl.read_parquet(self.lag_parquet)
                test_data = pl.read_parquet(self.test_parquet)
            else:
                raise ValueError("lag_parquet or test_parquet is None")
            for test_batch, validation_data in self.generate_data_batches(
                test_data, lag_data
            ):
                test, lags = test_batch
                if lags is None:
                    lags = pl.DataFrame()
                self.predict(test, lags)
        else:
            import kaggle_evaluation.jane_street_inference_server as js_server

            inference_server = js_server.JSInferenceServer((self.predict,))

            if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
                inference_server.serve()
            else:
                inference_server.run_local_gateway(
                    (self.test_parquet, self.lag_parquet)
                )

        self.pbar.close()

    def predict(
        self,
        test: pl.DataFrame,
        lags: pl.DataFrame,
    ) -> pl.DataFrame | pd.DataFrame:

        if lags is not None:
            self.lags_ = lags

        self.test_ = test

        if test["is_scored"].any():
            try:
                data_dict = defaultdict(list)
                for (symbol_id,), batch in test.group_by(
                    "symbol_id", maintain_order=True
                ):
                    if symbol_id in self.cache_history:
                        np.concatenate(
                            (
                                self.cache_history[symbol_id],
                                batch[self.feature_cols].to_numpy(),
                            ),
                            axis=0,
                        )
                    else:
                        self.cache_history[symbol_id] = batch[
                            self.feature_cols
                        ].to_numpy()
                    if len(self.cache_history[symbol_id]) > self.cache_lb_days:
                        self.cache_history[symbol_id] = self.cache_history[symbol_id][
                            -self.cache_lb_days :
                        ]

                    x = self.cache_history[symbol_id][-self.cache_lb_days :]
                    data_dict["x"].append(x)
                    data_dict["symbol_id"].append(symbol_id)
                    data_dict["row_id"].append(batch["row_id"].to_numpy())

                x_stack = np.stack(data_dict["x"])
                predicts = self.model.get_estimates(
                    x_stack, self.time_step_count, "responder_6"
                )
                predictions = pl.DataFrame(
                    {"row_id": data_dict["row_id"], "responder_6": predicts}
                )
            except Exception as e:
                print(f"Error: {e}")
        else:
            predictions = pl.DataFrame({"row_id": test["row_id"], "responder_6": 0})

        return predictions


if __name__ == "__main__":

    LOCAL_TEST = True
    KAGGLE_TEST = False

    model = EnsembleTimeSeriesV1(
        online_feature=OnlineMovingAverageCalculator(window=10),
        long_term_feature=ExpWeightedMeanCalculator(halflife=0.35, lookback=15),
        rev_decay_calculator=RevDecayCalculator(lookback=15),
        st_window=15,
        lt_window=15,
    )
    preprocessor = Preprocessor(
        symbol_id=None,
        responder=6,
        partition_ids=None,
        feature_set=None,
        sample_frequency=15,
        exclude_set=[
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
    )

    if LOCAL_TEST:
        inference_server = InferenceServer(
            model=model,
            preprocessor=preprocessor,
            dir=LOCAL_DATA_DIR,
            test_parquet=f"{LOCAL_DATA_DIR}/synthetic_test.parquet",
            lag_parquet=f"{LOCAL_DATA_DIR}/synthetic_lag.parquet",
            cache_lb_days=15,
            feature_cols=["responder_6"],
            test=True,  # Set to True for local testing
            partition_ids=[0],  # Specify partition IDs for synthetic data
            synthetic_days=50,  # Pass synthetic_days parameter
            num_score_dates=5,  # Pass num_score_dates parameter
        )
    elif KAGGLE_TEST:
        inference_server = InferenceServer(
            model=model,
            preprocessor=preprocessor,
            dir=DATA_DIR,
            test_parquet=f"{DATA_DIR}/synthetic_test.parquet",
            lag_parquet=f"{DATA_DIR}/synthetic_lag.parquet",
            cache_lb_days=15,
            feature_cols=["responder_6"],
            test=True,  # Set to False for Kaggle test
            partition_ids=[0],  # Specify partition IDs for synthetic data
            synthetic_days=50,  # Pass synthetic_days parameter
            num_score_dates=5,  # Pass num_score_dates parameter
        )
    else:
        inference_server = InferenceServer(
            model=model,
            preprocessor=preprocessor,
            dir=DATA_DIR,
            test_parquet=f"{DATA_DIR}/test.parquet",
            lag_parquet=f"{DATA_DIR}/lags.parquet",
            cache_lb_days=15,
            feature_cols=["responder_6"],
            test=False,  # Set to False for Kaggle submission
        )

    inference_server.run_inference_server()
