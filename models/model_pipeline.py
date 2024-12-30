import sys
from pathlib import Path

import pandas as pd

# Define a global variable for the project path
PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_PATH))

from scripts.feature_factory import (
    ExpWeightedMeanCalculator,
    FeatureFactory,
    MovingAverageCalculator,
    OnlineMovingAverageCalculator,
)
from scripts.preprocessing import Preprocessor


class ModelPipeline:

    def __init__(self, preprocessor: Preprocessor, feature_factory: FeatureFactory):
        self.preprocessor = preprocessor
        self.feature_factory = feature_factory

    def run(self, df, symbol_id, tdate):
        # Preprocess the data
        df = self.preprocessor.read_partition()

        # Create features
        df = self.feature_factory.calculate(df, tdate, "responder_6")

        # Make predictions
        predictions = df[df["date_id"] == tdate][["row_id", "responder_6"]]

        return predictions


if __name__ == "__main__":

    preprocessor = Preprocessor(
        responder=6,
        sample_frequency=5,
    )

    factory = FeatureFactory(
        calculators=[
            ExpWeightedMeanCalculator(halflife=0.35, lookback=15),
            OnlineMovingAverageCalculator(window=15),
        ],
        alpha=0.5,
    )
    pipeline = ModelPipeline(preprocessor=preprocessor, feature_factory=factory)

    # Example usage
    df = pd.DataFrame()  # Replace with actual DataFrame
    symbol_id = 0
    tdate = 100
    predictions = pipeline.run(df, symbol_id, tdate)
    print(predictions)
