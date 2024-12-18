import pandas as pd


class Preprocessor:
    def __init__(self):
        pass

    def filter_symbol(self, df, symbol):
        return df[df["symbol"] == symbol]

    def create_time_index(self, df):
        df["time_index"] = df["date_id"] * 968 + df["time_id"]
        return df

    def preprocess(self, df, symbol):
        df = self.filter_symbol(df, symbol)
        df = self.create_time_index(df)
        return df
