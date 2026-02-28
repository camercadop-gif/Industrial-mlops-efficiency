# industrial-mlops-efficiency/src/drift_detection.py

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from pathlib import Path


DATA_PATH = "data/raw/process_data.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("timestamp")
    return df


def temporal_reference_split(df):

    n = len(df)
    reference = df.iloc[:int(n * 0.7)]
    production = df.iloc[int(n * 0.85):]

    return reference, production


def calculate_psi(expected, actual, bins=10):

    expected_percents, bin_edges = np.histogram(expected, bins=bins)
    actual_percents, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    psi = np.sum(
        (actual_percents - expected_percents)
        * np.log((actual_percents + 1e-6) / (expected_percents + 1e-6))
    )

    return psi


def detect_drift(reference, production):

    results = {}

    features = ["temperature", "pressure", "flow", "composition", "efficiency"]

    for col in features:

        ks_stat, p_value = ks_2samp(reference[col], production[col])
        psi_value = calculate_psi(reference[col], production[col])

        results[col] = {
            "KS_statistic": float(ks_stat),
            "p_value": float(p_value),
            "PSI": float(psi_value)
        }

    return results


def main():

    df = load_data()
    reference, production = temporal_reference_split(df)

    results = detect_drift(reference, production)

    Path("results").mkdir(exist_ok=True)

    output_path = "results/drift_report.json"

    pd.DataFrame(results).T.to_json(output_path, indent=4)

    print("Drift analysis completed.")
    print(pd.DataFrame(results).T)


if __name__ == "__main__":
    main()