import sys
from pathlib import Path
from typing import IO, List, Optional

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# Define a global variable for the project path
DATA_DIR = Path("/kaggle/input/jane-street-real-time-market-data-forecasting")
PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_PATH))

import kaggle_evaluation.jane_street_inference_server as js_server
from scripts.calculators import (
    ExpWeightedMeanCalculator,
    MovingAverageCalculator,
    OnlineMovingAverageCalculator,
    RevDecayCalculator,
)
from scripts.models import BaseModel, EnsembleTimeSeriesV1
from scripts.preprocessing import Preprocessor


class Predictor:

    def __init__(
        self,
        model: BaseModel,
        preprocessor: Preprocessor,
        test_parquet: str | Path,
        lag_parquet: str | Path,
        cache_lb_days: int,
        feature_cols: List[str],
        pbar_length: int = 0,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.test_parquet = test_parquet
        self.lag_parquet = lag_parquet
        self.cache_lb_days = cache_lb_days
        self.feature_cols = feature_cols
        self.cache_history = {}
        self.test_ = None
        self.lags_ = None
        self.time_step_count = 0
        self.pbar = tqdm(total=pbar_length, disable=(pbar_length == 0))

    def run_inference_server(self):
        self.pbar.refresh()
        inference_server = js_server.JSInferenceServer(self.predict)

    def predict(
        self,
        test: pd.DataFrame | pl.DataFrame,
        lags: pd.DataFrame | pl.DataFrame,
    ) -> pl.DataFrame | pd.DataFrame:

        self.test_ = test
        if lags is not None:
            self.lags_ = lags

        n_symbols = test["symbol_id"].n_unique()

        priors = np.zeros((n_symbols, self.cache_lb_days, len(self.feature_cols)))
        row_ids = np.zeros(len(test), dtype=np.int64)

        test_partition = test.partition_by("symbol_id", as_dict=True)

        for i, ((symbol_id,), partition) in enumerate(test_partition.items()):
            if symbol_id in self.cache_history:
                self.cache_history[symbol_id] = np.concatenate(
                    (
                        self.cache_history[symbol_id],
                        partition[self.feature_cols].to_numpy(),
                    ),
                    axis=0,
                )
            else:
                self.cache_history[symbol_id] = partition[self.feature_cols].to_numpy()

            if len(self.cache_history[symbol_id]) > self.cache_lb_days:
                self.cache_history[symbol_id] = self.cache_history[symbol_id][
                    -self.cache_lb_days :
                ]

            x = self.cache_history[symbol_id][-self.cache_lb_days :]
            priors[i, -len(x) :, :] = x
            row_ids[i] = partition["row_id"].item()

        priors = np.nan_to_num(priors, nan=0.0)

        predicts = model.get_estimates(priors, tdate, "responder_6")

        predictions = pl.DataFrame(
            {"row_id": row_ids, "responder_6": predicts}
        ).with_columns(pl.col("row_id").cast(test["row_id"].dtype))

        # Join the predictions back to the test data for correct alignment
        predictions = (
            test.select("row_id")
            .join(predictions, on="row_id", how="left")
            .select("row_id", "responder_6")
        )

        # sanity check
        assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
        assert list(predictions.columns) == ["row_id", "responder_6"]
        assert len(predictions) == len(test)

        # update time_step_count
        self.time_step_count += 1
        self.pbar.update(1)

        return predictions


if __name__ == "__main__":

    TESTING = True

    if TESTING:
        syn_dir = "/kaggle/input/js24-rmf-generate-synthetic-test-data"
        test_parquet = f"{syn_dir}/synthetic_test.parquet"
        lag_parquet = f"{syn_dir}/synthetic_lag.parquet"
        total_time_steps = (
            pl.scan_parquet(test_parquet)
            .select((pl.col("date_id") * 10000 + pl.col("time_id")).n_unique())
            .collect()
            .item()
        )
    else:
        test_parquet = DATA_DIR / f"test.parquet"
        lag_parquet = DATA_DIR / f"lags.parquet"
        total_time_steps = 1

    preprocessor = Preprocessor(
        responder=6,
        sample_frequency=15,
    )

    CONFIG = {
        "symbol_id": None,
        "all_symbols": True,
        "responder": 6,
        "partition_ids": [0, 1, 2],
        "feature_set": [],
        "sample_frequency": 15,
        "exclude_set": [
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
        "total_lbs": 30,
    }

    model = EnsembleTimeSeriesV1(
        context="symbol_00",
        online_feature=OnlineMovingAverageCalculator(window=15),
        long_term_feature=ExpWeightedMeanCalculator(halflife=0.35, lookback=15),
        rev_decay_calculator=RevDecayCalculator(lookback=15),
        st_window=15,
        lt_window=15,
    )

    predictor = Predictor(
        model=model,
        preprocessor=preprocessor,
        test_parquet=test_parquet,
        lag_parquet=lag_parquet,
        cache_lb_days=CONFIG["total_lbs"],
        feature_cols=CONFIG["feature_set"],
        pbar_length=100,
    )

    data = Preprocessor().read_partition()
    predictor.predict
