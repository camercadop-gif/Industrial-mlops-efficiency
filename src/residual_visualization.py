# Residual Drift Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor

# ---------------------------
# Load latest trained model
# ---------------------------
model_files = sorted(Path("models").glob("*.json"))

if not model_files:
    raise FileNotFoundError("No trained model found inside /models folder.")

MODEL_PATH = model_files[-1]

model = XGBRegressor()
model.load_model(str(MODEL_PATH))

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("data/raw/process_data.csv").sort_values("timestamp")

# Temporal split
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
test_df = df.iloc[val_end:]

X_train = train_df[["temperature", "pressure", "flow", "composition"]]
y_train = train_df["efficiency"]

X_test = test_df[["temperature", "pressure", "flow", "composition"]]
y_test = test_df["efficiency"]

# ---------------------------
# Compute residuals
# ---------------------------
train_residuals = y_train - model.predict(X_train)
test_residuals = y_test - model.predict(X_test)

# ---------------------------
# Plot
# ---------------------------
plt.figure()
plt.hist(train_residuals, bins=30, alpha=0.5, density=True)
plt.hist(test_residuals, bins=30, alpha=0.5, density=True)

plt.title("Residual Distribution Comparison (Train vs Production)")
plt.xlabel("Residual Value")
plt.ylabel("Density")
plt.legend(["Train Residuals", "Production Residuals"])

plt.tight_layout()

# Save inside project root
plt.savefig("residual_drift_visualization.png")

plt.show()

Path("reports").mkdir(exist_ok=True)
plt.savefig("reports/residual_drift_visualization.png")
