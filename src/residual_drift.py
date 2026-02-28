# industrial-mlops-efficiency/src/residual_drift.py

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp
from xgboost import XGBRegressor


DATA_PATH = "data/raw/process_data.csv"
MODEL_PATH = sorted(Path("models").glob("model_v1_*.json"))[-1]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("timestamp")
    return df


def temporal_split(df):

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    test = df.iloc[val_end:]

    return train, test


def prepare_features(df):
    X = df[["temperature", "pressure", "flow", "composition"]]
    y = df["efficiency"]
    return X, y


def load_model():
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def compute_residuals(model, X, y):
    preds = model.predict(X)
    residuals = y - preds
    return residuals


def calculate_psi(expected, actual, bins=10):

    expected_counts, bin_edges = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    psi = np.sum(
        (actual_percents - expected_percents)
        * np.log((actual_percents + 1e-6) / (expected_percents + 1e-6))
    )

    return psi


def main():

    df = load_data()
    train_df, test_df = temporal_split(df)

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    model = load_model()

    train_residuals = compute_residuals(model, X_train, y_train)
    test_residuals = compute_residuals(model, X_test, y_test)

    ks_stat, p_value = ks_2samp(train_residuals, test_residuals)
    psi_value = calculate_psi(train_residuals, test_residuals)

    print("Residual Drift Analysis")
    print("------------------------")
    print(f"KS Statistic: {ks_stat}")
    print(f"p-value: {p_value}")
    print(f"PSI: {psi_value}")

    if p_value < 0.05:
        print("\nDrift detected in residual distribution.")
    else:
        print("\nNo significant residual drift detected.")


if __name__ == "__main__":
    main()