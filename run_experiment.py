#!/usr/bin/env python3
"""
run_experiment.py
=================
End-to-end experiment runner.

Usage:
    python run_experiment.py [--target homo|lumo|gap|mu|alpha|zpve]
                             [--n_samples 300]
                             [--n_qubits 4]
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── allow running from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as CFG
from data.loader import download_qm9_sample, load_and_describe
from preprocessing.features import MolecularFeaturizer
from models.classical import ClassicalModels
from models.hybrid import HybridQuantumClassical, build_hybrid_suite
from quantum.circuit import QuantumFeatureMap
from evaluation.metrics import compare_models
from evaluation.visualize import (
    plot_model_comparison, plot_predictions, plot_training_time,
    plot_circuit_diagram, plot_feature_importance, plot_summary_dashboard,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target",    default=CFG.TARGET)
    p.add_argument("--n_samples", type=int, default=CFG.N_SAMPLES)
    p.add_argument("--n_qubits",  type=int, default=CFG.N_QUBITS)
    p.add_argument("--n_layers",  type=int, default=CFG.N_Q_LAYERS)
    p.add_argument("--no_plots",  action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    TARGET = args.target
    print(f"\n{'='*65}")
    print(f"  Quantum-Enhanced Molecular Property Prediction")
    print(f"  Target: {TARGET}  |  Samples: {args.n_samples}  |  Qubits: {args.n_qubits}")
    print(f"{'='*65}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("► Step 1: Loading dataset ...")
    df = download_qm9_sample(
        save_path=os.path.join(CFG.DATA_DIR, "qm9_sample.csv"),
        n_samples=args.n_samples,
        random_seed=CFG.RANDOM_SEED,
    )
    load_and_describe(df)

    y = df[TARGET].values
    smiles = df["smiles"].tolist()

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n► Step 2: Molecular featurization (RDKit) ...")
    feat = MolecularFeaturizer(
        n_bits=CFG.MORGAN_BITS,
        radius=CFG.MORGAN_RADIUS,
        use_pca=CFG.USE_PCA,
        n_pca_components=CFG.N_PCA,
    )
    X, valid_idx = feat.fit_transform(smiles)
    y = y[valid_idx]
    print(f"  Feature matrix: {X.shape}  |  Targets: {y.shape}")

    # Quantum sub-features (angle encoded, top-k dimensions)
    X_q, top_idx = feat.quantum_features(X, n_qubits=args.n_qubits)
    print(f"  Quantum feature shape: {X_q.shape}  (top features: {top_idx})")

    # ── 3. Train / test split ─────────────────────────────────────────────────
    print("\n► Step 3: Train/test split ...")
    (X_tr, X_te, Xq_tr, Xq_te, y_tr, y_te) = train_test_split(
        X, X_q, y,
        test_size=CFG.TEST_SIZE,
        random_state=CFG.RANDOM_SEED,
    )
    print(f"  Train: {len(y_tr)}  |  Test: {len(y_te)}")

    # ── 4. Classical models ───────────────────────────────────────────────────
    print("\n► Step 4: Training classical models ...")
    cm = ClassicalModels(random_state=CFG.RANDOM_SEED)
    classical_results = cm.fit_evaluate_all(X_tr, y_tr, X_te, y_te)
    best_c_name, _ = cm.get_best("rmse")

    # ── 5. Quantum feature map ────────────────────────────────────────────────
    print("\n► Step 5: Building quantum feature map ...")
    qfm = QuantumFeatureMap(n_qubits=args.n_qubits, n_layers=args.n_layers,
                            random_state=CFG.RANDOM_SEED)
    print(f"  Circuit info:\n{qfm.circuit_diagram()}")

    # ── 6. Hybrid models ──────────────────────────────────────────────────────
    print("\n► Step 6: Training hybrid quantum-classical models ...")
    hybrid_suite = build_hybrid_suite(qfm, random_state=CFG.RANDOM_SEED)
    hybrid_results = {}
    for name, hmodel in hybrid_suite.items():
        hmodel.fit(X_tr, Xq_tr, y_tr)
        res = hmodel.evaluate(X_te, Xq_te, y_te)
        hybrid_results[name] = res
        print(f"  {name:30s}  MAE={res['mae']:.4f}  RMSE={res['rmse']:.4f}  "
              f"R²={res['r2']:.4f}  t={res['train_time']:.2f}s")

    # ── 7. Comparison table ───────────────────────────────────────────────────
    print("\n► Step 7: Results summary")
    comp_df = compare_models(classical_results, hybrid_results)
    print("\n" + comp_df.to_string(index=False))

    best_hybrid_name = comp_df[comp_df["Type"] == "Hybrid Q-C"].iloc[0]["Model"]
    best_hybrid_pred = hybrid_results[best_hybrid_name]["y_pred"]
    best_classical_pred = classical_results[best_c_name]["y_pred"]

    # Save CSV
    csv_path = os.path.join(CFG.RESULTS_DIR, f"results_{TARGET}.csv")
    comp_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved → {csv_path}")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n► Step 8: Generating plots ...")
        R = CFG.RESULTS_DIR

        plot_model_comparison(comp_df, TARGET,
            save_path=os.path.join(R, f"model_comparison_{TARGET}.png"))

        pred_dict = {best_c_name: best_classical_pred,
                     best_hybrid_name: best_hybrid_pred}
        plot_predictions(y_te, pred_dict, TARGET,
            save_path=os.path.join(R, f"predictions_{TARGET}.png"))

        plot_training_time(comp_df,
            save_path=os.path.join(R, "training_time.png"))

        plot_circuit_diagram(
            save_path=os.path.join(R, "circuit_diagram.png"))

        plot_feature_importance(feat, X_te, y_te,
            save_path=os.path.join(R, "feature_importance.png"))

        plot_summary_dashboard(
            comp_df, y_te, best_classical_pred, best_hybrid_pred,
            target_name=TARGET,
            save_path=os.path.join(R, f"dashboard_{TARGET}.png"),
        )
        print(f"  All plots saved to {R}/")

    print(f"\n{'='*65}")
    print("  Experiment complete!")
    print(f"{'='*65}\n")
    return comp_df, classical_results, hybrid_results


if __name__ == "__main__":
    main()
