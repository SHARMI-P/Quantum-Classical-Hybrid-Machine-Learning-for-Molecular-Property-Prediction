# models/classical_models.py — Classical Baseline Models
"""
Classical machine learning baselines for molecular property prediction.

Models:
1. Ridge Regression — linear baseline, interpretable
2. Random Forest — strong non-linear baseline, no feature scaling needed
3. MLP (Scikit-learn) — multi-layer perceptron, moderate complexity
4. GradientBoosting — typically best classical model on tabular data

All models are wrapped in a consistent interface with:
  - fit(X_train, y_train)
  - predict(X_test) → y_pred
  - get_complexity() → dict with param count, model size
"""

import os
import sys
import time
import logging
import numpy as np
import pickle
from typing import Dict, Any, Optional, Tuple

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RF_N_ESTIMATORS, RF_MAX_DEPTH, MLP_HIDDEN_LAYERS, MLP_MAX_ITER, RANDOM_STATE

logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all models with consistent interface."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.train_time = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseModel":
        logger.info(f"Training {self.name}...")
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - t0
        self.is_fitted = True
        logger.info(f"  {self.name} trained in {self.train_time:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted, f"{self.name} not trained yet"
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Return dict of evaluation metrics."""
        y_pred = self.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "train_time": self.train_time or 0.0,
        }

    def get_complexity(self) -> Dict[str, Any]:
        """Return model complexity metrics."""
        raise NotImplementedError

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


class RidgeModel(BaseModel):
    """L2-regularized linear regression — linear baseline."""

    def __init__(self, alpha: float = 1.0):
        super().__init__("Ridge Regression")
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=RANDOM_STATE if hasattr(Ridge, 'random_state') else None))
        ])
        self.alpha = alpha

    def get_complexity(self) -> Dict[str, Any]:
        if not self.is_fitted:
            return {}
        coef = self.model.named_steps["ridge"].coef_
        return {
            "n_parameters": len(coef) + 1,  # +1 for intercept
            "model_type": "linear",
            "is_linear": True,
        }


class RandomForestModel(BaseModel):
    """Random Forest — strong non-linear baseline."""

    def __init__(
        self,
        n_estimators: int = RF_N_ESTIMATORS,
        max_depth: int = RF_MAX_DEPTH,
    ):
        super().__init__("Random Forest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        self.n_estimators = n_estimators

    def get_complexity(self) -> Dict[str, Any]:
        if not self.is_fitted:
            return {}
        total_nodes = sum(
            tree.tree_.node_count
            for tree in self.model.estimators_
        )
        return {
            "n_estimators": self.n_estimators,
            "total_nodes": total_nodes,
            "n_parameters": total_nodes * 3,  # threshold + feature + value
            "model_type": "ensemble",
            "is_linear": False,
        }

    def feature_importances(self, feature_names=None) -> Dict[str, float]:
        """Return top feature importances."""
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]
        idx = np.argsort(importances)[::-1][:20]
        return {feature_names[i]: importances[i] for i in idx}


class MLPModel(BaseModel):
    """Multi-layer perceptron (scikit-learn)."""

    def __init__(
        self,
        hidden_layer_sizes: tuple = MLP_HIDDEN_LAYERS,
        max_iter: int = MLP_MAX_ITER,
    ):
        super().__init__("MLP Neural Network")
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                random_state=RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False,
            ))
        ])
        self.hidden_layer_sizes = hidden_layer_sizes

    def get_complexity(self) -> Dict[str, Any]:
        if not self.is_fitted:
            return {}
        mlp = self.model.named_steps["mlp"]
        n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
        return {
            "architecture": self.hidden_layer_sizes,
            "n_parameters": n_params,
            "n_layers": len(self.hidden_layer_sizes) + 1,
            "model_type": "neural_network",
            "is_linear": False,
        }

    def get_training_curve(self) -> Optional[np.ndarray]:
        """Return validation loss curve if available."""
        mlp = self.model.named_steps["mlp"]
        if hasattr(mlp, "loss_curve_"):
            return np.array(mlp.loss_curve_)
        return None


class GradientBoostingModel(BaseModel):
    """Gradient Boosting — often best classical model on tabular data."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
    ):
        super().__init__("Gradient Boosting")
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=RANDOM_STATE,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )

    def get_complexity(self) -> Dict[str, Any]:
        if not self.is_fitted:
            return {}
        return {
            "n_estimators": self.model.n_estimators_,
            "n_parameters": self.model.n_estimators_ * (2 ** self.model.max_depth) * 3,
            "model_type": "gradient_boosting",
            "is_linear": False,
        }


def train_all_classical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, BaseModel]:
    """
    Train all classical baseline models and return them.
    
    Returns:
        dict mapping model_name → fitted model
    """
    models = {
        "ridge": RidgeModel(alpha=10.0),
        "random_forest": RandomForestModel(),
        "mlp": MLPModel(),
        "gradient_boosting": GradientBoostingModel(),
    }

    results = {}
    print("\n" + "=" * 60)
    print("TRAINING CLASSICAL BASELINE MODELS")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_val, y_val)
        results[name] = model

        print(f"\n{model.name}:")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  Time: {metrics['train_time']:.2f}s")
        complexity = model.get_complexity()
        if "n_parameters" in complexity:
            print(f"  Params: {complexity['n_parameters']:,}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    # Synthetic test
    X = np.random.randn(500, 50)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + np.random.randn(500) * 0.1

    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    models = train_all_classical(X_train, y_train, X_test, y_test)

    print("\n─── Test Set Evaluation ───")
    for name, model in models.items():
        metrics = model.evaluate(X_test, y_test)
        print(f"{model.name}: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
