import matplotlib.pyplot as plt
import pandas as pd
import polars as pl


def plot_time_series(df, column, title):
    plt.figure(figsize=(12, 6))
    plt.plot(df["time_index"], df[column])
    plt.xlabel("Time")
    plt.ylabel(column)
    plt.title(title)
    plt.show()


def get_correlations(df, columns):
    return df[columns].corr()


if __name__ == "__main__":

    plot_time_series(df, "feature_0", "Feature 0 Time Series")
