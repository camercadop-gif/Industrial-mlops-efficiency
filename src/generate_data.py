# industrial-mlops-efficiency/src/generate_data.py

import numpy as np
import pandas as pd
from pathlib import Path


def generate_dataset(n_samples=5000, random_state=42):

    np.random.seed(random_state)

    timestamp = np.arange(n_samples)

    temperature = np.random.uniform(300, 500, n_samples)
    pressure = np.random.uniform(1, 10, n_samples)
    flow = np.random.uniform(10, 100, n_samples)
    composition = np.random.uniform(0, 1, n_samples)

    degradation = 0.00005 * timestamp

    efficiency = np.zeros(n_samples)

    for i in range(n_samples):

        # -------- BEFORE DRIFT --------
        if timestamp[i] < 0.8 * n_samples:

            temp_effect = np.exp(-((temperature[i] - 400) ** 2) / 2000)
            pressure_penalty = -0.02 * (pressure[i] - 5) ** 2
            flow_interaction = 0.0005 * temperature[i] * flow[i]
            composition_effect = 1 / (1 + np.exp(-10 * (composition[i] - 0.5)))

        # -------- AFTER STRUCTURAL DRIFT --------
        else:

            # Temperature optimum shifts
            temp_effect = np.exp(-((temperature[i] - 430) ** 2) / 1500)

            # Pressure sensitivity increases
            pressure_penalty = -0.04 * (pressure[i] - 6) ** 2

            # Flow interaction changes magnitude
            flow_interaction = 0.0002 * temperature[i] * flow[i]

            # Composition becomes more dominant
            composition_effect = 1 / (1 + np.exp(-15 * (composition[i] - 0.6)))

        efficiency[i] = (
            0.5 * temp_effect
            + pressure_penalty
            + flow_interaction
            + 0.3 * composition_effect
            - degradation[i]
        )

    noise = np.random.normal(0, 0.02, n_samples)
    efficiency = efficiency + noise

    efficiency = (efficiency - efficiency.min()) / (
        efficiency.max() - efficiency.min()
    )

    df = pd.DataFrame({
        "timestamp": timestamp,
        "temperature": temperature,
        "pressure": pressure,
        "flow": flow,
        "composition": composition,
        "efficiency": efficiency
    })

    return df


def save_dataset(df):

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/process_data.csv", index=False)

    print("Dataset generated successfully.")
    print("Latent variable NOT saved (simulating real industrial condition).")

    print(f"Dataset saved at: data/raw/process_data.csv")


def main():

    df = generate_dataset()
    save_dataset(df)


if __name__ == "__main__":
    main()