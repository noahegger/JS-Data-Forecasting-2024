from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_data(
    train_path: str, output_dir: str, start_date: int, end_date: int
):
    train_df = pd.read_parquet(train_path)

    # Filter data based on start_date and end_date
    synthetic_df = train_df[
        (train_df["date_id"] >= start_date) & (train_df["date_id"] <= end_date)
    ]

    # Generate synthetic test data
    test_df = synthetic_df.copy()
    test_df["target"] = np.nan  # Remove target values for test data

    # Generate synthetic lag data
    lag_df = synthetic_df.copy()

    # Save synthetic data
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(f"{output_dir}/synthetic_test.parquet")
    lag_df.to_parquet(f"{output_dir}/synthetic_lag.parquet")


if __name__ == "__main__":
    generate_synthetic_data(
        train_path="/path/to/train.parquet",
        output_dir="/path/to/output",
        start_date=0,
        end_date=100,
    )
