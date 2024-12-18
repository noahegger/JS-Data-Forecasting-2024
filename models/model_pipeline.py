from scripts.analysis import Analyzer
from scripts.feature_factory import FeatureFactory, MovingAverageCalculator
from scripts.preprocessing import Preprocessor


class ModelPipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.analyzer = Analyzer()
        self.feature_factory = FeatureFactory([MovingAverageCalculator(window=5)])

    def run(self, df, symbol):
        # Preprocess the data
        df = self.preprocessor.preprocess(df, symbol)

        # Create features
        df = self.feature_factory.create_features(df)

        # Analyze the data
        self.analyzer.plot_time_series(df, "responder_6", "Responder 6 Time Series")
        self.analyzer.plot_time_series(
            df, "moving_average_5", "5-Day Moving Average of Responder 6"
        )

        return df
