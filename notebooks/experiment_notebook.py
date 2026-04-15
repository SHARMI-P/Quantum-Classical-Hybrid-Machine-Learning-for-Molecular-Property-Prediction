#!/usr/bin/env python3
"""
notebooks/experiment_notebook.py
==================================
Step-by-step walkthrough of the full pipeline.
Run this as a script or copy cells into a Jupyter notebook.
"""

# ─── Cell 1: Imports ──────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ─── Cell 2: Load Data ────────────────────────────────────────────────────────
from data.loader import download_qm9_sample, load_and_describe

df = download_qm9_sample(save_path="../data/qm9_sample.csv", n_samples=200)
load_and_describe(df)
print(df.head())

# ─── Cell 3: Feature Engineering ─────────────────────────────────────────────
from preprocessing.features import MolecularFeaturizer

TARGET = "homo"
feat = MolecularFeaturizer(n_bits=128, radius=2, use_pca=True, n_pca_components=10)
X, valid_idx = feat.fit_transform(df["smiles"].tolist())
y = df[TARGET].values[valid_idx]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target ({TARGET}) stats: mean={y.mean():.3f}, std={y.std():.3f}")

# ─── Cell 4: Quantum Features ─────────────────────────────────────────────────
X_q, top_idx = feat.quantum_features(X, n_qubits=4)
print(f"\nQuantum-ready features shape: {X_q.shape}")
print(f"Top feature indices selected: {top_idx}")
print(f"Feature range: [{X_q.min():.3f}, {X_q.max():.3f}] (angle encoding in [0, π])")

# ─── Cell 5: Train/Test Split ─────────────────────────────────────────────────
X_tr, X_te, Xq_tr, Xq_te, y_tr, y_te = train_test_split(
    X, X_q, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(y_tr)} | Test: {len(y_te)}")

# ─── Cell 6: Classical Baselines ─────────────────────────────────────────────
from models.classical import ClassicalModels

cm = ClassicalModels(random_state=42)
c_results = cm.fit_evaluate_all(X_tr, y_tr, X_te, y_te)
print("\nClassical Model Summary:")
print(cm.summary_df().to_string(index=False))

# ─── Cell 7: Quantum Circuit ──────────────────────────────────────────────────
from quantum.circuit import QuantumFeatureMap

qfm = QuantumFeatureMap(n_qubits=4, n_layers=2, random_state=42)
print(f"\nCircuit diagram:\n{qfm.circuit_diagram()}")

# Test on a single sample
sample_q = qfm.transform(Xq_te[:1])
print(f"\nQuantum output for one molecule: {sample_q}")

# ─── Cell 8: Hybrid Model ─────────────────────────────────────────────────────
from models.hybrid import HybridQuantumClassical
from sklearn.ensemble import GradientBoostingRegressor

hybrid = HybridQuantumClassical(
    qfm, GradientBoostingRegressor(n_estimators=80, random_state=42)
)
hybrid.fit(X_tr, Xq_tr, y_tr)
h_res = hybrid.evaluate(X_te, Xq_te, y_te)
print(f"\nHybrid model: MAE={h_res['mae']:.4f}  RMSE={h_res['rmse']:.4f}  R²={h_res['r2']:.4f}")

# ─── Cell 9: Comparison ───────────────────────────────────────────────────────
from evaluation.metrics import compare_models

hybrid_results = {"Hybrid (GBT)": h_res}
comp = compare_models(c_results, hybrid_results)
print("\nFull Comparison:")
print(comp.to_string(index=False))

# ─── Cell 10: Quick Plot ──────────────────────────────────────────────────────
from evaluation.visualize import plot_summary_dashboard

best_c = cm.results["Gradient Boosting"]["y_pred"]
plot_summary_dashboard(comp, y_te, best_c, h_res["y_pred"],
                       target_name=TARGET, save_path="../results/notebook_dashboard.png")
print("Plot saved to ../results/notebook_dashboard.png")
