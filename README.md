# Jane Street Data Forecasting Competition - 2024

This repository contains my implementation for the Jane Street Real-Time Market Data Forecasting competition hosted on Kaggle. Although my participation began late, and my submission was not fully optimized within the competition timeframe, I developed a comprehensive local testing pipeline. This pipeline emulates the Kaggle evaluation API closely, enabling rapid local iteration and validation.

## Competition Overview

Jane Street's competition involved predicting real-world financial market data, emphasizing the complexity of accurately modeling financial instruments. Challenges included handling fat-tailed distributions, non-stationary time series, and sudden shifts in market behavior. The provided dataset was anonymized to maintain proprietary confidentiality, yet it captured essential features of real trading environments.

The goal was to predict `responder_6`, assessed through a weighted zero-mean R-squared scoring metric:

$$
R^2 = 1 - \frac{\sum_i w_i (y_i - \hat{y}_i)^2}{\sum_i w_i y_i^2}
$$
where $y_i$ and $\hat{y}_i$ are the actual and predicted values, and $w_i$ are sample weights.

## Repository Structure

The project was structured to support modularity, maintainability, and reproducibility:

- `data_preprocessing.py`: Scripts for reading, resampling, and filtering datasets.
- `utils.py`: Utility functions for weighted R-squared scoring and time-weighted calculations.
- `models.py`: Implementation of interpretable models including linear (Lasso and Ridge regressions) and a blended time series model.
- `record.py`: Custom data structures for symbol-specific caching and correlation analyses.
- `calculators.py`: Advanced calculators for feature transformation, including median, z-score normalization, and exponential weighted mean.
- `plotting.py`: Comprehensive visualization utilities for correlation analysis, time-series exploration, and performance tracking.
- `analysis.py`: High-level scripts for correlation analyses and feature exploration.

## Modeling Approach

I believe in prioritizing the simplest approaches and only layering on additional complexity when justified. Given the noisy nature of financial data, overfitting is a constant hazard. My approach prioritized:

- **Interpretability**: Focusing on linear models such as Lasso and Ridge regression, as well as simple time series approaches which provided insights into feature relevance.
- **Feature Engineering**: Generating and evaluating potential features through exhaustive correlation tests, rolling statistics, and feature interaction terms. The anonymized data prevents any domain-specific knowledge, so feature engineering required some creativity in regard to how base features may be related. With time series, first approaches warrant cumulative/non-cumulative differencing, summing, and experimentation with sign.
- **Correlation Analysis**: Systematic assessment of potential correlations using Spearman and Pearson with magnitude cutoffs.

## Local Testing Pipeline

The major strength of my implementation relative to others' was the sophisticated local testing environment designed to mimic Kaggle's evaluation server. This system enabled:

- **Rapid Iteration**: Quick experimentation with customizable, synthetic datasets and immediate feedback through custom R² scoring --- both for single- and full-instrument portfolios.
- **Synthetic Data Generation**: Methods to create and customize synthetic data mirroring the competition's format, allowing safe experimentation without overfitting.
- **Performance Monitoring**: Real-time tracking of model predictions for single and multiple instruments, ground truths, cumulative and non-cumulative errors, and correlation metrics.

## Visualization and Analysis

The `plotting.py` module supported extensive visual diagnostics for:

- Idea Generation
- Performance Evaluation

This allowed for quickly tracking model behavior and exploring feature relationships with the target variable. Some specifics included:

- Rolling correlations for various features.
- Feature distribution analysis and autocorrelation functions.
- Comparative visualizations of predicted versus true values.

## Environment Setup

The project used a conda environment (`environment.yaml`) with dependencies clearly specified, ensuring replicability:

```bash
conda env create -f environment.yaml
conda activate js_env

