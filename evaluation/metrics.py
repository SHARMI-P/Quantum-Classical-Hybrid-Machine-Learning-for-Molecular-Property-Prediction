"""
evaluation/metrics.py
=====================
Unified evaluation helpers and result aggregation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred, model_name="Model", train_time=0.0):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  [{model_name:30s}]  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  t={train_time:.2f}s")
    return {"model": model_name, "mae": mae, "rmse": rmse, "r2": r2, "train_time": train_time}


def compare_models(classical_results: dict, hybrid_results: dict) -> pd.DataFrame:
    rows = []
    for name, res in classical_results.items():
        rows.append({
            "Model": name, "Type": "Classical",
            "MAE": round(res["mae"], 4), "RMSE": round(res["rmse"], 4),
            "R²": round(res["r2"], 4), "Train Time (s)": round(res["train_time"], 3),
        })
    for name, res in hybrid_results.items():
        rows.append({
            "Model": name, "Type": "Hybrid Q-C",
            "MAE": round(res["mae"], 4), "RMSE": round(res["rmse"], 4),
            "R²": round(res["r2"], 4), "Train Time (s)": round(res["train_time"], 3),
        })
    df = pd.DataFrame(rows).sort_values("RMSE")
    return df
