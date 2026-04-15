"""
models/hybrid.py
================
Hybrid Quantum-Classical model.

Pipeline:
  1. Classical feature extraction (already done upstream)
  2. Select top-k features → quantum angle encoding
  3. Run variational quantum circuit → expectation values
  4. Concatenate quantum features with remaining classical features
  5. Train a classical regressor head on the combined representation

This follows the "data re-uploading" / quantum kernel paradigm.
"""

import time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class HybridQuantumClassical:
    """
    Hybrid Quantum-Classical Regressor.

    Parameters
    ----------
    quantum_feature_map : QuantumFeatureMap instance
    classical_head      : sklearn-compatible regressor
    n_classical_extra   : how many extra classical features to append
                          alongside quantum output
    """

    def __init__(self, quantum_feature_map, classical_head=None, n_classical_extra=8):
        self.qfm = quantum_feature_map
        self.n_qubits = quantum_feature_map.n_qubits
        self.n_classical_extra = n_classical_extra
        self.head = classical_head or Ridge(alpha=0.5)
        self._train_time = None

    # ------------------------------------------------------------------
    def _make_hybrid_features(self, X_classical, X_quantum_angles):
        """
        quantum circuit outputs + top classical features → combined vector.
        """
        q_out = self.qfm.transform(X_quantum_angles)           # (n, n_qubits)
        # Append some raw classical features for extra signal
        n_extra = min(self.n_classical_extra, X_classical.shape[1])
        X_extra = X_classical[:, :n_extra]
        return np.hstack([q_out, X_extra])                     # (n, n_qubits + n_extra)

    def fit(self, X_classical, X_quantum_angles, y):
        print(f"  [hybrid] Building quantum features ({X_quantum_angles.shape[0]} samples) ...",
              end=" ", flush=True)
        t0 = time.perf_counter()
        X_hybrid = self._make_hybrid_features(X_classical, X_quantum_angles)
        print(f"hybrid shape={X_hybrid.shape}", end=" | ", flush=True)
        self.head.fit(X_hybrid, y)
        self._train_time = time.perf_counter() - t0
        print(f"head fitted  t={self._train_time:.2f}s")
        return self

    def predict(self, X_classical, X_quantum_angles):
        X_hybrid = self._make_hybrid_features(X_classical, X_quantum_angles)
        return self.head.predict(X_hybrid)

    def evaluate(self, X_classical, X_quantum_angles, y_true):
        y_pred = self.predict(X_classical, X_quantum_angles)
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        return {
            "mae": mae, "rmse": rmse, "r2": r2,
            "train_time": self._train_time or 0.0,
            "y_pred": y_pred,
        }


# ── Helper: build three hybrid variants with different heads ─────────────────
def build_hybrid_suite(qfm, random_state=42):
    rs = random_state
    return {
        "Hybrid (Ridge)": HybridQuantumClassical(
            qfm, Ridge(alpha=0.5)),
        "Hybrid (RF)": HybridQuantumClassical(
            qfm, RandomForestRegressor(n_estimators=50, random_state=rs)),
        "Hybrid (GBT)": HybridQuantumClassical(
            qfm, GradientBoostingRegressor(n_estimators=80, learning_rate=0.1,
                                           random_state=rs)),
    }
