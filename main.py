# main.py — Main Pipeline Entry Point
"""
Quantum-Enhanced Molecular Property Prediction
Full end-to-end pipeline execution.

Usage:
    python main.py                         # Run with default config
    python main.py --n_molecules 500       # Smaller run
    python main.py --target mu             # Different property
    python main.py --no_quantum            # Classical only (faster)
    python main.py --encoding amplitude    # Try amplitude encoding
"""

import os
import sys
import time
import logging
import argparse
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─── Setup paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import (
    N_MOLECULES, TARGET_PROPERTY, N_QUBITS, N_LAYERS,
    HYBRID_EPOCHS, RESULTS_DIR, PROPERTY_INFO, RANDOM_STATE
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(RESULTS_DIR, "experiment.log")),
    ]
)
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_STATE)


def print_banner():
    print("\n" + "═" * 70)
    print("  ⚛  QUANTUM-ENHANCED MOLECULAR PROPERTY PREDICTION")
    print("     Hybrid Quantum-Classical Machine Learning Pipeline")
    print("═" * 70)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_molecules", type=int, default=N_MOLECULES)
    parser.add_argument("--target", type=str, default=TARGET_PROPERTY,
                        choices=list(PROPERTY_INFO.keys()))
    parser.add_argument("--no_quantum", action="store_true")
    parser.add_argument("--encoding", type=str, default="angle",
                        choices=["angle", "amplitude"])
    parser.add_argument("--n_qubits", type=int, default=N_QUBITS)
    parser.add_argument("--n_layers", type=int, default=N_LAYERS)
    parser.add_argument("--epochs", type=int, default=HYBRID_EPOCHS)
    return parser.parse_args()


def run_pipeline(args=None):
    """Execute the full experiment pipeline."""
    print_banner()
    t_start = time.time()

    if args is None:
        args = parse_args()

    target_info = PROPERTY_INFO[args.target]
    logger.info(f"Target property: {target_info['name']} ({args.target})")
    logger.info(f"Dataset size: {args.n_molecules} molecules")
    logger.info(f"Quantum: {'disabled' if args.no_quantum else f'{args.n_qubits} qubits, {args.n_layers} layers'}")

    # ─── Step 1: Load Dataset ─────────────────────────────────────────────────
    logger.info("\n" + "─" * 60)
    logger.info("STEP 1: Loading Dataset")
    logger.info("─" * 60)

    from data.loader import load_dataset
    df = load_dataset(n_molecules=args.n_molecules)
    logger.info(f"Loaded {len(df)} molecules")
    logger.info(f"Target column: {args.target}")
    logger.info(f"Target stats: mean={df[args.target].mean():.3f}, "
                f"std={df[args.target].std():.3f}")

    # ─── Step 2: Preprocessing ────────────────────────────────────────────────
    logger.info("\n" + "─" * 60)
    logger.info("STEP 2: Feature Extraction & Preprocessing")
    logger.info("─" * 60)

    from preprocessing.pipeline import MolecularPreprocessor
    preprocessor = MolecularPreprocessor(n_components=args.n_qubits)
    splits_pca, splits_raw = preprocessor.fit_transform(df, target_col=args.target)
    preprocessor.summary()

    X_train_pca = splits_pca["X_train"]
    X_val_pca   = splits_pca["X_val"]
    X_test_pca  = splits_pca["X_test"]
    y_train     = splits_pca["y_train"]
    y_val       = splits_pca["y_val"]
    y_test      = splits_pca["y_test"]

    X_train_raw = splits_raw["X_train"]
    X_val_raw   = splits_raw["X_val"]
    X_test_raw  = splits_raw["X_test"]

    # ─── Step 3: Classical Models ─────────────────────────────────────────────
    logger.info("\n" + "─" * 60)
    logger.info("STEP 3: Training Classical Baseline Models")
    logger.info("─" * 60)

    from models.classical_models import (
        RidgeModel, RandomForestModel, MLPModel, GradientBoostingModel
    )

    classical_models = {
        "ridge": RidgeModel(alpha=10.0),
        "random_forest": RandomForestModel(),
        "gradient_boosting": GradientBoostingModel(n_estimators=150),
        "mlp": MLPModel(),
    }

    # Classical models use raw (non-PCA) features for better performance
    for name, model in classical_models.items():
        model.fit(X_train_raw, y_train)

    # ─── Step 4: Quantum / Hybrid Models ─────────────────────────────────────
    results = {}
    predictions = {}
    train_losses = {}
    val_losses = {}

    if not args.no_quantum:
        logger.info("\n" + "─" * 60)
        logger.info("STEP 4: Training Hybrid Quantum-Classical Models")
        logger.info("─" * 60)

        try:
            from models.hybrid_model import QuantumClassicalHybrid, HybridNeuralNetWrapper

            # 4a: QFM + Ridge (angle encoding)
            logger.info(f"\n[4a] Quantum Feature Map + Ridge (angle encoding)")
            qch_angle = QuantumClassicalHybrid(
                n_qubits=args.n_qubits,
                n_layers=args.n_layers,
                encoding="angle",
                classical_head="ridge",
                optimize_circuit=True,
            )
            qch_angle.fit(X_train_pca, y_train, n_optimization_steps=20)
            train_losses["hybrid_vqc"] = qch_angle.train_losses

            # 4b: Hybrid Neural Net (if PyTorch + PennyLane available)
            try:
                logger.info(f"\n[4b] Hybrid Neural Network (VQC layer)")
                hnn = HybridNeuralNetWrapper(
                    input_dim=args.n_qubits,
                    n_qubits=args.n_qubits,
                    n_layers=max(1, args.n_layers - 1),
                    epochs=args.epochs,
                )
                hnn.fit(X_train_pca, y_train, X_val_pca, y_val)
                train_losses["hybrid_nn"] = hnn.train_losses
                val_losses["hybrid_nn"] = hnn.val_losses

                results["hybrid_nn"] = hnn.evaluate(X_test_pca, y_test)
                predictions["hybrid_nn"] = hnn.predict(X_test_pca)
            except Exception as e:
                logger.warning(f"HybridNN failed: {e}")

            results["hybrid_vqc"] = qch_angle.evaluate(X_test_pca, y_test)
            predictions["hybrid_vqc"] = qch_angle.predict(X_test_pca)

        except Exception as e:
            logger.warning(f"Quantum models failed: {e}")

    # ─── Step 5: Evaluate All Models ─────────────────────────────────────────
    logger.info("\n" + "─" * 60)
    logger.info("STEP 5: Evaluation")
    logger.info("─" * 60)

    # Evaluate classical models (using raw features)
    for name, model in classical_models.items():
        results[name] = model.evaluate(X_test_raw, y_test)
        predictions[name] = model.predict(X_test_raw)

    # Print results table
    print("\n" + "═" * 65)
    print(f"  RESULTS — {target_info['name']} ({target_info['unit']})")
    print("═" * 65)
    print(f"  {'Model':<28} {'MAE':>8} {'RMSE':>8} {'R²':>7} {'Time':>8}")
    print("─" * 65)

    sorted_models = sorted(results.keys(), key=lambda m: results[m]["mae"])
    for m in sorted_models:
        r = results[m]
        t = r.get("train_time", 0)
        t_str = f"{t:.1f}s" if t >= 1 else f"{t*1000:.0f}ms"
        tag = "⚛" if "hybrid" in m else " "
        label = {"ridge": "Ridge Regression",
                  "random_forest": "Random Forest",
                  "gradient_boosting": "Gradient Boosting",
                  "mlp": "MLP Neural Network",
                  "hybrid_vqc": "Hybrid VQC (angle)",
                  "hybrid_nn": "Hybrid Neural Net"}.get(m, m)
        print(f"  {tag} {label:<26} {r['mae']:>8.4f} {r['rmse']:>8.4f} "
              f"{r['r2']:>7.4f} {t_str:>8}")

    print("═" * 65)

    # Best model
    best_model = sorted_models[0]
    print(f"\n  ★ Best model: {best_model.upper()} (MAE = {results[best_model]['mae']:.4f})")

    # Classical vs quantum comparison
    classical_maes = {m: results[m]["mae"] for m in sorted_models
                       if "hybrid" not in m and "quantum" not in m}
    quantum_maes = {m: results[m]["mae"] for m in sorted_models
                     if "hybrid" in m or "quantum" in m}

    if classical_maes and quantum_maes:
        best_classical = min(classical_maes.values())
        best_quantum = min(quantum_maes.values())
        improvement = (best_classical - best_quantum) / best_classical * 100
        print(f"\n  Classical best: MAE = {best_classical:.4f}")
        print(f"  Quantum best:   MAE = {best_quantum:.4f}")
        if improvement > 0:
            print(f"  → Quantum improvement: {improvement:.1f}%")
        else:
            print(f"  → Classical advantage: {-improvement:.1f}% (quantum overhead for small data)")

    # ─── Step 6: Visualizations ───────────────────────────────────────────────
    logger.info("\n" + "─" * 60)
    logger.info("STEP 6: Generating Visualizations")
    logger.info("─" * 60)

    try:
        from evaluation.visualizer import generate_all_plots
        saved_plots = generate_all_plots(
            results=results,
            predictions=predictions,
            y_test=y_test,
            explained_variance_ratio=preprocessor.explained_variance_ratio,
            train_losses=train_losses if train_losses else None,
            val_losses=val_losses if val_losses else None,
            target_info=target_info,
        )
        print(f"\n  Generated {len(saved_plots)} plots in: {RESULTS_DIR}/")
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    # ─── Step 7: Save Results ─────────────────────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, "results.json")
    serializable = {}
    for m, r in results.items():
        serializable[m] = {k: float(v) for k, v in r.items()}

    with open(results_path, "w") as f:
        json.dump({
            "target": args.target,
            "n_molecules": args.n_molecules,
            "n_qubits": args.n_qubits,
            "results": serializable,
        }, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n  Total experiment time: {total_time:.1f}s")
    print(f"  Results saved to: {results_path}")
    print("═" * 70 + "\n")

    return results, predictions, y_test


if __name__ == "__main__":
    run_pipeline()
