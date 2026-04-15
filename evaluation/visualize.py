"""
evaluation/visualize.py
========================
All plotting functions for the project.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os

PALETTE = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "border":  "#30363d",
    "accent1": "#58a6ff",
    "accent2": "#3fb950",
    "accent3": "#f78166",
    "accent4": "#d2a8ff",
    "text":    "#e6edf3",
    "subtext": "#8b949e",
}

CLASSICAL_COLOR = "#58a6ff"
HYBRID_COLOR    = "#3fb950"

def _setup_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["surface"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["subtext"],
        "ytick.color":       PALETTE["subtext"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["border"],
        "grid.alpha":        0.5,
        "font.family":       "monospace",
        "figure.dpi":        120,
    })


# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(comparison_df, target_name="homo", save_path=None):
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Model Comparison  ·  Target: {target_name}",
                 fontsize=14, color=PALETTE["text"], y=1.02)

    metrics = ["MAE", "RMSE", "R²"]
    for ax, metric in zip(axes, metrics):
        colors = [HYBRID_COLOR if "Hybrid" in t else CLASSICAL_COLOR
                  for t in comparison_df["Type"]]
        vals = comparison_df[metric].values
        bars = ax.barh(comparison_df["Model"], vals, color=colors, alpha=0.85)
        ax.set_xlabel(metric, color=PALETTE["text"])
        ax.set_title(metric, color=PALETTE["accent1"], fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=7, color=PALETTE["subtext"])
        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(color=CLASSICAL_COLOR, label="Classical"),
                   Patch(color=HYBRID_COLOR,    label="Hybrid Q-C")]
        ax.legend(handles=handles, loc="lower right", fontsize=7,
                  facecolor=PALETTE["bg"], edgecolor=PALETTE["border"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[viz] Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_predictions(y_test, pred_dict, target_name="homo", save_path=None):
    """Scatter: actual vs predicted for each model."""
    _setup_style()
    n = len(pred_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Predicted vs Actual  ·  {target_name}",
                 fontsize=13, color=PALETTE["text"])

    for ax, (name, y_pred) in zip(axes, pred_dict.items()):
        color = HYBRID_COLOR if "Hybrid" in name else CLASSICAL_COLOR
        ax.scatter(y_test, y_pred, alpha=0.4, s=15, color=color)
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color=PALETTE["accent3"], lw=1.5)
        ax.set_xlabel("Actual", color=PALETTE["text"])
        ax.set_ylabel("Predicted", color=PALETTE["text"])
        ax.set_title(name, color=PALETTE["accent1"], fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[viz] Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_training_time(comparison_df, save_path=None):
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [HYBRID_COLOR if "Hybrid" in t else CLASSICAL_COLOR
              for t in comparison_df["Type"]]
    bars = ax.bar(comparison_df["Model"], comparison_df["Train Time (s)"],
                  color=colors, alpha=0.85)
    ax.set_ylabel("Training Time (s)", color=PALETTE["text"])
    ax.set_title("Training Time Comparison", color=PALETTE["accent1"], fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}s", ha="center", fontsize=8,
                color=PALETTE["subtext"])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[viz] Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_circuit_diagram(save_path=None):
    """ASCII-art style circuit diagram."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["bg"])

    diagram = """
 ┌─────────────────────────────────────────────────────────────────────────┐
 │              Variational Quantum Circuit (4 qubits, 2 layers)           │
 ├─────────────────────────────────────────────────────────────────────────┤
 │                                                                         │
 │  x₀ ──┤ RY(x₀) ├──●──────────────────┤ Rot(θ₀) ├──●──────────── ⟨Z₀⟩ │
 │                    │                              │                     │
 │  x₁ ──┤ RY(x₁) ├──X──●───────────────┤ Rot(θ₁) ├──X──●──────── ⟨Z₁⟩ │
 │                       │                              │                  │
 │  x₂ ──┤ RY(x₂) ├─────X──●────────────┤ Rot(θ₂) ├─────X──●──── ⟨Z₂⟩ │
 │                          │                               │              │
 │  x₃ ──┤ RY(x₃) ├────────X────────────┤ Rot(θ₃) ├───────X──── ⟨Z₃⟩  │
 │                                                                         │
 │  [Encoding Layer]    [Entangling Layer 1]   [Entangling Layer 2]        │
 │  Angle Encoding      CNOT cascade          CNOT cascade + Rot gates     │
 └─────────────────────────────────────────────────────────────────────────┘
"""
    ax.text(0.05, 0.5, diagram, transform=ax.transAxes,
            fontsize=11, color=PALETTE["accent2"],
            fontfamily="monospace", va="center")
    ax.set_title("Quantum Circuit Architecture",
                 color=PALETTE["accent1"], fontsize=14, pad=10)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[viz] Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(featurizer, X, y, save_path=None):
    """Show PCA variance explained and descriptor correlation with target."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Feature Analysis", color=PALETTE["text"], fontsize=13)

    # PCA explained variance
    if featurizer.pca is not None:
        ev = featurizer.pca.explained_variance_ratio_
        axes[0].bar(range(len(ev)), ev * 100,
                    color=PALETTE["accent4"], alpha=0.8)
        axes[0].set_xlabel("PCA Component", color=PALETTE["text"])
        axes[0].set_ylabel("Variance Explained (%)", color=PALETTE["text"])
        axes[0].set_title("PCA Explained Variance", color=PALETTE["accent1"])
        axes[0].grid(alpha=0.3)

    # Feature-target correlation
    corr = np.abs(np.corrcoef(X[:, :20].T, y)[:-1, -1])
    axes[1].bar(range(len(corr)), corr,
                color=PALETTE["accent1"], alpha=0.8)
    axes[1].set_xlabel("Feature Index (top 20)", color=PALETTE["text"])
    axes[1].set_ylabel("|Pearson r| with Target", color=PALETTE["text"])
    axes[1].set_title("Feature-Target Correlation", color=PALETTE["accent1"])
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[viz] Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_dashboard(comparison_df, y_test, best_classical_pred,
                           best_hybrid_pred, target_name="homo", save_path=None):
    """A single 2×3 dashboard combining all key visuals."""
    _setup_style()
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "Quantum-Enhanced Molecular Property Prediction  ·  Results Dashboard",
        fontsize=16, color=PALETTE["text"], y=0.98, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # 1. RMSE comparison bar
    ax1 = fig.add_subplot(gs[0, 0])
    colors = [HYBRID_COLOR if "Hybrid" in t else CLASSICAL_COLOR
              for t in comparison_df["Type"]]
    ax1.barh(comparison_df["Model"], comparison_df["RMSE"],
             color=colors, alpha=0.85)
    ax1.set_xlabel("RMSE", color=PALETTE["text"])
    ax1.set_title("RMSE by Model", color=PALETTE["accent1"], fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # 2. R² comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(comparison_df["Model"], comparison_df["R²"],
             color=colors, alpha=0.85)
    ax2.set_xlabel("R²", color=PALETTE["text"])
    ax2.set_title("R² Score by Model", color=PALETTE["accent1"], fontweight="bold")
    ax2.axvline(0, color=PALETTE["border"], lw=1)
    ax2.grid(axis="x", alpha=0.3)

    # 3. Training time
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(comparison_df["Model"], comparison_df["Train Time (s)"],
            color=colors, alpha=0.85)
    ax3.set_ylabel("Time (s)", color=PALETTE["text"])
    ax3.set_title("Training Time", color=PALETTE["accent1"], fontweight="bold")
    ax3.tick_params(axis="x", rotation=40, labelsize=7)
    ax3.grid(axis="y", alpha=0.3)

    # 4. Best classical scatter
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(y_test, best_classical_pred, alpha=0.4, s=15,
                color=CLASSICAL_COLOR)
    lo, hi = y_test.min(), y_test.max()
    ax4.plot([lo, hi], [lo, hi], "--", color=PALETTE["accent3"], lw=1.5)
    ax4.set_xlabel("Actual", color=PALETTE["text"])
    ax4.set_ylabel("Predicted", color=PALETTE["text"])
    ax4.set_title("Best Classical: Actual vs Pred", color=PALETTE["accent1"])
    ax4.grid(alpha=0.3)

    # 5. Best hybrid scatter
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(y_test, best_hybrid_pred, alpha=0.4, s=15,
                color=HYBRID_COLOR)
    ax5.plot([lo, hi], [lo, hi], "--", color=PALETTE["accent3"], lw=1.5)
    ax5.set_xlabel("Actual", color=PALETTE["text"])
    ax5.set_ylabel("Predicted", color=PALETTE["text"])
    ax5.set_title("Best Hybrid Q-C: Actual vs Pred", color=PALETTE["accent1"])
    ax5.grid(alpha=0.3)

    # 6. Residuals comparison
    ax6 = fig.add_subplot(gs[1, 2])
    resid_c = y_test - best_classical_pred
    resid_h = y_test - best_hybrid_pred
    ax6.hist(resid_c, bins=25, alpha=0.6, color=CLASSICAL_COLOR, label="Classical")
    ax6.hist(resid_h, bins=25, alpha=0.6, color=HYBRID_COLOR,    label="Hybrid Q-C")
    ax6.axvline(0, color=PALETTE["accent3"], lw=1.5, ls="--")
    ax6.set_xlabel("Residual", color=PALETTE["text"])
    ax6.set_ylabel("Count", color=PALETTE["text"])
    ax6.set_title("Residual Distribution", color=PALETTE["accent1"])
    ax6.legend(fontsize=8, facecolor=PALETTE["bg"])
    ax6.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=PALETTE["bg"])
        print(f"[viz] Saved → {save_path}")
    return fig
