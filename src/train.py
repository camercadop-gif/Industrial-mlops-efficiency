# industrial-mlops-efficiency/src/train.py

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor



DATA_PATH = "data/raw/process_data.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("timestamp")
    return df


def temporal_split(df):

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


def prepare_features(df):
    X = df[["temperature", "pressure", "flow", "composition"]]
    y = df["efficiency"]
    return X, y


def evaluate_model(model, X, y):

    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


def save_feature_importance(model, feature_names):

    importances = model.feature_importances_

    plt.figure()
    plt.bar(feature_names, importances)
    plt.xticks(rotation=45)
    plt.title("Feature Importance")
    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/feature_importance.png")
    plt.close()


def main():

    df = load_data()

    train_df, val_df, test_df = temporal_split(df)

    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    model = XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    ) 

    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"model_v1_{timestamp}"

    Path("models").mkdir(exist_ok=True)
    model.save_model(f"models/{model_version}.json")

    metrics = {
        "model_version": model_version,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics
    }

    Path("results").mkdir(exist_ok=True)
    with open(f"results/metrics_{model_version}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    save_feature_importance(model, X_train.columns)

    print("Training completed.")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()