"""
models/classical.py
===================
Classical ML baselines for molecular property prediction.

Models:
  - LinearRegression   (interpretable baseline)
  - RandomForestRegressor (ensemble, handles non-linearity)
  - MLPRegressor        (small neural network)
"""

import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ClassicalModels:
    """
    Train and evaluate a suite of classical regression models.

    Usage
    -----
    cm = ClassicalModels()
    results = cm.fit_evaluate_all(X_train, y_train, X_test, y_test)
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = self._build_models()
        self.trained = {}
        self.results = {}

    def _build_models(self):
        rs = self.random_state
        return {
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=rs, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=rs),
            "Neural Network": MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu",
                max_iter=500, random_state=rs, early_stopping=True),
        }

    # ------------------------------------------------------------------
    def fit_evaluate_all(self, X_train, y_train, X_test, y_test):
        """
        Train every model, record metrics and training time.

        Returns
        -------
        dict  {model_name: {mae, rmse, r2, train_time, y_pred}}
        """
        for name, model in self.models.items():
            print(f"  [classical] Training {name} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0

            y_pred = model.predict(X_test)
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2   = r2_score(y_test, y_pred)

            self.trained[name] = model
            self.results[name] = {
                "mae": mae, "rmse": rmse, "r2": r2,
                "train_time": elapsed, "y_pred": y_pred,
            }
            print(f"MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  t={elapsed:.2f}s")

        return self.results

    def get_best(self, metric="rmse"):
        """Return (name, model) with the lowest RMSE (or chosen metric)."""
        best = min(self.results, key=lambda k: self.results[k][metric])
        return best, self.trained[best]

    def summary_df(self):
        import pandas as pd
        rows = []
        for name, res in self.results.items():
            rows.append({
                "Model": name,
                "MAE": round(res["mae"], 4),
                "RMSE": round(res["rmse"], 4),
                "R²": round(res["r2"], 4),
                "Train Time (s)": round(res["train_time"], 3),
            })
        return pd.DataFrame(rows).sort_values("RMSE")
