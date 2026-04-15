# evaluation/visualizer.py — Comprehensive Visualization Suite
"""
All plotting functions for the quantum-classical comparison project.

Generates:
1. Model performance comparison (bar chart: MAE, RMSE, R²)
2. Predicted vs Actual scatter plots (all models)
3. Training time comparison
4. Training curves (loss over epochs)
5. PCA explained variance
6. Feature importance (Random Forest)
7. Quantum circuit diagram (ASCII)
8. Model complexity comparison
9. Error distribution (violin plots)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Any
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, PROPERTY_INFO

logger = logging.getLogger(__name__)

# ─── Style Configuration ──────────────────────────────────────────────────────
COLORS = {
    "ridge": "#64748b",
    "random_forest": "#2563eb",
    "gradient_boosting": "#7c3aed",
    "mlp": "#0891b2",
    "hybrid_vqc": "#dc2626",
    "hybrid_nn": "#ea580c",
    "quantum": "#dc2626",
    "classical": "#2563eb",
}

MODEL_LABELS = {
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "gradient_boosting": "Grad. Boosting",
    "mlp": "MLP (Classical)",
    "hybrid_vqc": "Hybrid VQC",
    "hybrid_nn": "Hybrid NN",
}

plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "legend.facecolor": "#1e293b",
    "legend.edgecolor": "#475569",
    "legend.labelcolor": "#e2e8f0",
})


def _save(fig, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def plot_performance_comparison(
    results: Dict[str, Dict[str, float]],
    target_name: str = "HOMO-LUMO Gap",
    filename: str = "01_performance_comparison.png",
):
    """
    Bar chart comparing MAE, RMSE, R² across all models.
    Classical models shown in blue tones, quantum in red/orange.
    """
    models = list(results.keys())
    mae_vals = [results[m]["mae"] for m in models]
    rmse_vals = [results[m]["rmse"] for m in models]
    r2_vals = [results[m]["r2"] for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    colors = [COLORS.get(m, "#64748b") for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f"Model Performance Comparison\nTarget: {target_name}",
        y=1.02, fontsize=16
    )

    metrics = [
        (axes[0], mae_vals,  "MAE (↓ better)", "Mean Absolute Error"),
        (axes[1], rmse_vals, "RMSE (↓ better)", "Root Mean Squared Error"),
        (axes[2], r2_vals,   "R² (↑ better)",  "R² Score"),
    ]

    for ax, vals, ylabel, title in metrics:
        bars = ax.barh(labels, vals, color=colors, edgecolor="#475569",
                       linewidth=0.5, height=0.6)

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(
                val + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center", ha="left",
                fontsize=9, color="#e2e8f0"
            )

        ax.set_title(title, pad=10)
        ax.set_xlabel(ylabel)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)

    # Highlight best model
    best_mae = min(models, key=lambda m: results[m]["mae"])
    fig.text(
        0.5, -0.03,
        f"★ Best model (MAE): {MODEL_LABELS.get(best_mae, best_mae)}",
        ha="center", fontsize=12, color="#fbbf24"
    )

    plt.tight_layout()
    return _save(fig, filename)


def plot_predicted_vs_actual(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    target_name: str = "HOMO-LUMO Gap (eV)",
    filename: str = "02_predicted_vs_actual.png",
):
    """Scatter plots: predicted vs actual for each model."""
    n_models = len(predictions)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_models == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    fig.suptitle(f"Predicted vs Actual — {target_name}", y=1.02)

    for i, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        color = COLORS.get(model_name, "#64748b")

        # Scatter
        ax.scatter(y_true, y_pred, alpha=0.5, s=15, c=color, edgecolors="none")

        # Perfect prediction line
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "w--", lw=1.5, alpha=0.6, label="Perfect")

        # R² annotation
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes,
                va="top", fontsize=10, color="#fbbf24",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", alpha=0.8))

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(MODEL_LABELS.get(model_name, model_name), color=color)
        ax.grid(alpha=0.3)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return _save(fig, filename)


def plot_training_time(
    results: Dict[str, Dict[str, float]],
    filename: str = "03_training_time.png",
):
    """Log-scale bar chart of training times."""
    models = list(results.keys())
    times = [results[m].get("train_time", 0) + 0.01 for m in models]  # +0.01 to avoid log(0)
    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = [COLORS.get(m, "#64748b") for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Training Time Comparison", y=1.02)

    bars = ax.barh(labels, times, color=colors, edgecolor="#475569", height=0.6)

    for bar, t in zip(bars, times):
        t_actual = t - 0.01
        label = f"{t_actual:.1f}s" if t_actual > 0.5 else f"{t_actual*1000:.0f}ms"
        ax.text(t * 1.05, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9, color="#e2e8f0")

    ax.set_xscale("log")
    ax.set_xlabel("Training Time (seconds, log scale)")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Add "quantum overhead" annotation
    classical_times = [results[m].get("train_time", 0) for m in models
                       if "quantum" not in m and "hybrid" not in m]
    quantum_times = [results[m].get("train_time", 0) for m in models
                     if "hybrid" in m or "quantum" in m]

    if classical_times and quantum_times:
        ratio = np.mean(quantum_times) / max(np.mean(classical_times), 0.01)
        fig.text(0.98, 0.02, f"Quantum overhead: ~{ratio:.1f}×",
                 ha="right", fontsize=10, color="#fbbf24", style="italic")

    plt.tight_layout()
    return _save(fig, filename)


def plot_training_curves(
    train_losses: Dict[str, List[float]],
    val_losses: Dict[str, List[float]] = None,
    filename: str = "04_training_curves.png",
):
    """Training and validation loss curves."""
    if not train_losses:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Training Curves", y=1.02)

    for model_name, losses in train_losses.items():
        if losses:
            color = COLORS.get(model_name, "#64748b")
            label = MODEL_LABELS.get(model_name, model_name)
            ax.plot(losses, color=color, lw=2, label=f"{label} (train)", alpha=0.9)

            if val_losses and model_name in val_losses and val_losses[model_name]:
                ax.plot(val_losses[model_name], color=color, lw=2,
                        linestyle="--", label=f"{label} (val)", alpha=0.6)

    ax.set_xlabel("Epoch / Optimization Step")
    ax.set_ylabel("Loss (MSE)")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    return _save(fig, filename)


def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    filename: str = "05_pca_variance.png",
):
    """PCA explained variance plot."""
    n_components = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PCA Dimensionality Reduction Analysis", y=1.02)

    # Per-component
    x = np.arange(1, n_components + 1)
    bars = ax1.bar(x, explained_variance_ratio * 100,
                   color="#2563eb", edgecolor="#475569", width=0.7)
    for bar, val in zip(bars, explained_variance_ratio):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8)

    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("Per-Component Variance")
    ax1.set_xticks(x)
    ax1.grid(axis="y", alpha=0.3)

    # Cumulative
    ax2.fill_between(x, cumulative * 100, alpha=0.3, color="#dc2626")
    ax2.plot(x, cumulative * 100, "o-", color="#dc2626", lw=2, markersize=8)

    # 90% threshold line
    ax2.axhline(y=90, color="#fbbf24", linestyle="--", alpha=0.7, label="90% threshold")
    ax2.axhline(y=cumulative[-1] * 100, color="#22d3ee", linestyle=":",
                alpha=0.7, label=f"Total: {cumulative[-1]*100:.1f}%")

    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_title("Cumulative Variance Retained")
    ax2.set_xticks(x)
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return _save(fig, filename)


def plot_error_distribution(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    filename: str = "06_error_distribution.png",
):
    """Violin plots of prediction errors across models."""
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    errors_data = []
    for model_name, y_pred in predictions.items():
        errors = y_pred - y_true
        label = MODEL_LABELS.get(model_name, model_name)
        errors_data.append((label, errors, COLORS.get(model_name, "#64748b")))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Prediction Error Distribution", y=1.02)

    positions = range(1, len(errors_data) + 1)

    for pos, (label, errors, color) in zip(positions, errors_data):
        parts = ax.violinplot([errors], positions=[pos], showmedians=True,
                               showquartiles=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor("#475569")
        parts["cmedians"].set_color("#fbbf24")
        parts["cmedians"].set_linewidth(2)
        if "cmins" in parts:
            parts["cmins"].set_color(color)
            parts["cmaxes"].set_color(color)
            parts["cbars"].set_color(color)

    ax.axhline(0, color="#e2e8f0", linestyle="--", alpha=0.5, label="Zero error")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([d[0] for d in errors_data], rotation=15, ha="right")
    ax.set_ylabel("Prediction Error")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return _save(fig, filename)


def plot_summary_dashboard(
    results: Dict[str, Dict[str, float]],
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    target_name: str = "HOMO-LUMO Gap (eV)",
    filename: str = "00_summary_dashboard.png",
):
    """
    Master summary dashboard combining all key metrics in one figure.
    """
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0f172a")

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.45, wspace=0.35,
        top=0.90, bottom=0.07, left=0.07, right=0.97
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.96,
        "Quantum-Enhanced Molecular Property Prediction",
        ha="center", fontsize=20, fontweight="bold", color="#f8fafc"
    )
    fig.text(
        0.5, 0.925,
        f"Target: {target_name}  |  Hybrid VQC vs Classical Baselines",
        ha="center", fontsize=12, color="#94a3b8"
    )

    models = list(results.keys())
    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = [COLORS.get(m, "#64748b") for m in models]

    # ── 1. MAE Comparison ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    mae_vals = [results[m]["mae"] for m in models]
    bars = ax1.barh(labels, mae_vals, color=colors, edgecolor="#334155", height=0.55)
    for bar, val in zip(bars, mae_vals):
        ax1.text(val + max(mae_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9, color="#e2e8f0")
    ax1.set_title("Mean Absolute Error (↓ Better)", color="#e2e8f0")
    ax1.set_xlabel("MAE")
    ax1.grid(axis="x", alpha=0.3)
    ax1.set_xlim(0, max(mae_vals) * 1.2)

    # ── 2. R² Comparison ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    r2_vals = [results[m]["r2"] for m in models]
    bars2 = ax2.barh(labels, r2_vals, color=colors, edgecolor="#334155", height=0.55)
    for bar, val in zip(bars2, r2_vals):
        ax2.text(min(val - 0.02, max(r2_vals) * 0.9), bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9, color="#e2e8f0", ha="right")
    ax2.set_title("R² Score (↑ Better)", color="#e2e8f0")
    ax2.set_xlabel("R²")
    ax2.set_xlim(0, 1.05)
    ax2.grid(axis="x", alpha=0.3)

    # ── 3–4. Predicted vs Actual (best classical + quantum) ──────────────────
    classical_models = [m for m in models if "hybrid" not in m and "quantum" not in m]
    quantum_models = [m for m in models if "hybrid" in m or "quantum" in m]

    for col, model_list, cat in [(0, classical_models, "Classical"), (1, quantum_models, "Quantum")]:
        if not model_list:
            continue
        best_m = min(model_list, key=lambda m: results[m]["mae"])
        ax = fig.add_subplot(gs[1, col * 2: col * 2 + 2])
        y_pred = predictions[best_m]
        c = COLORS.get(best_m, "#64748b")
        ax.scatter(y_test, y_pred, alpha=0.5, s=12, c=c, edgecolors="none")
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        m = (hi - lo) * 0.05
        ax.plot([lo - m, hi + m], [lo - m, hi + m], "w--", lw=1.5, alpha=0.5)
        r2 = np.corrcoef(y_test, y_pred)[0, 1] ** 2
        ax.text(0.05, 0.93, f"R² = {r2:.3f}", transform=ax.transAxes,
                va="top", fontsize=10, color="#fbbf24",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.8))
        ax.set_title(f"Best {cat}: {MODEL_LABELS.get(best_m, best_m)}", color=c)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)

    # ── 5. Training Time ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    times = [max(results[m].get("train_time", 0.01), 0.01) for m in models]
    bars5 = ax5.bar(labels, times, color=colors, edgecolor="#334155", width=0.6)
    for bar, t in zip(bars5, times):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f"{t:.1f}s" if t >= 1 else f"{t*1000:.0f}ms",
                 ha="center", va="bottom", fontsize=9, color="#e2e8f0")
    ax5.set_yscale("log")
    ax5.set_title("Training Time (log scale)", color="#e2e8f0")
    ax5.set_ylabel("Seconds (log)")
    ax5.tick_params(axis="x", rotation=20)
    ax5.grid(axis="y", alpha=0.3)

    # ── 6. Metrics table ─────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis("off")

    table_data = [["Model", "MAE", "RMSE", "R²", "Time"]]
    for m in models:
        r = results[m]
        t = r.get("train_time", 0)
        t_str = f"{t:.1f}s" if t >= 1 else f"{t*1000:.0f}ms"
        table_data.append([
            MODEL_LABELS.get(m, m),
            f"{r['mae']:.4f}",
            f"{r['rmse']:.4f}",
            f"{r['r2']:.3f}",
            t_str,
        ])

    table = ax6.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style table
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("#1e293b" if row % 2 == 0 else "#0f172a")
        cell.set_text_props(color="#e2e8f0")
        cell.set_edgecolor("#334155")
        if row == 0:
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(color="#93c5fd", fontweight="bold")

    ax6.set_title("Results Summary", color="#e2e8f0", pad=15)

    return _save(fig, filename)


def plot_circuit_diagram(filename: str = "07_circuit_diagram.png"):
    """Generate a visual circuit diagram."""
    try:
        import pennylane as qml
        n_qubits = 4
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x, params):
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
            for layer in range(2):
                for i in range(n_qubits):
                    qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        x = np.random.randn(n_qubits)
        params = np.random.randn(2, n_qubits, 3)

        fig, ax = plt.subplots(figsize=(14, 5))
        qml.draw_mpl(circuit, decimals=2)(x, params)
        plt.title("Variational Quantum Circuit (4 qubits, 2 layers)", color="#e2e8f0")
        return _save(plt.gcf(), filename)

    except Exception as e:
        logger.warning(f"Circuit diagram failed: {e}")
        # Draw a schematic instead
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle("VQC Architecture Schematic")
        ax.axis("off")

        n_qubits = 4
        for i in range(n_qubits):
            y = 0.8 - i * 0.2
            ax.annotate("", xy=(0.95, y), xytext=(0.05, y),
                        arrowprops=dict(arrowstyle="-", color="#e2e8f0", lw=1.5))
            ax.text(0.02, y, f"|0⟩", va="center", fontsize=12, color="#60a5fa")

            # Encoding gate
            rect = FancyBboxPatch((0.12, y - 0.04), 0.08, 0.08,
                                   boxstyle="round,pad=0.01",
                                   facecolor="#1e3a5f", edgecolor="#60a5fa", lw=1.5)
            ax.add_patch(rect)
            ax.text(0.16, y, f"RY(x{i})", va="center", ha="center",
                    fontsize=8, color="#93c5fd")

            # Ansatz gates
            for layer in range(2):
                x_pos = 0.3 + layer * 0.25
                rect = FancyBboxPatch((x_pos, y - 0.04), 0.10, 0.08,
                                       boxstyle="round,pad=0.01",
                                       facecolor="#3b1d5f", edgecolor="#a78bfa", lw=1.5)
                ax.add_patch(rect)
                ax.text(x_pos + 0.05, y, f"Rot", va="center", ha="center",
                        fontsize=8, color="#c4b5fd")

            # Measurement
            rect = FancyBboxPatch((0.86, y - 0.04), 0.08, 0.08,
                                   boxstyle="round,pad=0.01",
                                   facecolor="#1f2d1f", edgecolor="#4ade80", lw=1.5)
            ax.add_patch(rect)
            ax.text(0.90, y, "⟨Z⟩", va="center", ha="center",
                    fontsize=10, color="#4ade80")

        ax.text(0.16, 0.92, "Encoding\n(Angle)", ha="center", fontsize=9, color="#93c5fd")
        ax.text(0.35, 0.92, "Layer 1\n(Ansatz)", ha="center", fontsize=9, color="#c4b5fd")
        ax.text(0.60, 0.92, "Layer 2\n(Ansatz)", ha="center", fontsize=9, color="#c4b5fd")
        ax.text(0.90, 0.92, "Measure", ha="center", fontsize=9, color="#4ade80")

        return _save(fig, filename)


def generate_all_plots(
    results: Dict[str, Dict[str, float]],
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    explained_variance_ratio: np.ndarray = None,
    train_losses: Dict[str, List] = None,
    val_losses: Dict[str, List] = None,
    target_info: Dict = None,
):
    """Generate all visualizations."""
    if target_info is None:
        target_info = {"name": "Property", "unit": ""}

    target_name = f"{target_info['name']} ({target_info['unit']})"

    saved = []
    logger.info("\nGenerating visualizations...")

    saved.append(plot_summary_dashboard(results, predictions, y_test, target_name))
    saved.append(plot_performance_comparison(results, target_name))
    saved.append(plot_predicted_vs_actual(predictions, y_test, target_name))
    saved.append(plot_training_time(results))
    saved.append(plot_error_distribution(predictions, y_test))

    if explained_variance_ratio is not None:
        saved.append(plot_pca_variance(explained_variance_ratio))

    if train_losses:
        saved.append(plot_training_curves(train_losses, val_losses))

    saved.append(plot_circuit_diagram())

    logger.info(f"Generated {len(saved)} plots in {RESULTS_DIR}/")
    return saved
