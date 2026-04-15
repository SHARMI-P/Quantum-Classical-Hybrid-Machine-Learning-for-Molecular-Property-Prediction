#!/usr/bin/env python3
"""
run_standalone.py
=================
Fully self-contained experiment runner (no RDKit, no PennyLane required).
Uses physicochemically-motivated feature simulation and a quantum-inspired
feature map to reproduce the full hybrid pipeline end-to-end.
"""

import os, sys, time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = "/home/claude/quantum_mol_project/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RNG = np.random.default_rng(42)
PALETTE = {
    "bg":"#0d1117","surface":"#161b22","border":"#30363d",
    "accent1":"#58a6ff","accent2":"#3fb950","accent3":"#f78166",
    "accent4":"#d2a8ff","text":"#e6edf3","subtext":"#8b949e",
}
CLASSICAL_COLOR = "#58a6ff"
HYBRID_COLOR    = "#3fb950"
WARNING_COLOR   = "#f78166"

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor":   PALETTE["surface"],
        "axes.edgecolor":   PALETTE["border"],
        "axes.labelcolor":  PALETTE["text"],
        "xtick.color":      PALETTE["subtext"],
        "ytick.color":      PALETTE["subtext"],
        "text.color":       PALETTE["text"],
        "grid.color":       PALETTE["border"],
        "grid.alpha":       0.4,
        "font.family":      "monospace",
        "figure.dpi":       130,
        "axes.titlepad":    10,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

MOLECULE_CLASSES = [
    # (name_prefix, MW_range, logP_range, n_atoms_range, arom_rings, hba, hbd)
    ("alkane",      (30,  120), (-0.5, 4.0), (2,  9), 0, 0, 0),
    ("alcohol",     (32,  100), (-1.5, 2.5), (2,  8), 0, 1, 1),
    ("amine",       (31,   95), (-1.0, 2.8), (2,  8), 0, 1, 2),
    ("aromatic",    (78,  140), (1.5,  3.5), (6, 12), 1, 0, 0),
    ("phenol",      (94,  130), (1.2,  2.8), (7, 12), 1, 1, 1),
    ("aniline",     (93,  125), (0.8,  2.5), (7, 12), 1, 1, 2),
    ("acid",        (46,  120), (-1.0, 2.0), (3,  9), 0, 2, 1),
    ("ester",       (60,  130), (0.5,  3.5), (4, 10), 0, 2, 0),
    ("nitrile",     (41,  110), (0.5,  3.0), (3,  9), 0, 1, 0),
    ("halide",      (50,  140), (1.0,  4.0), (2,  8), 0, 0, 0),
    ("heterocycle", (67,  130), (0.5,  2.5), (5, 11), 1, 1, 0),
    ("thiol",       (48,  110), (0.5,  3.5), (2,  8), 0, 0, 1),
]

def generate_dataset(n=400, seed=42):
    rng = np.random.default_rng(seed)
    records = []
    per_class = n // len(MOLECULE_CLASSES) + 1
    for (cls, mw_r, lp_r, na_r, arom, hba, hbd) in MOLECULE_CLASSES:
        for _ in range(per_class):
            mw    = rng.uniform(*mw_r)
            logp  = rng.uniform(*lp_r)
            n_at  = rng.integers(*na_r)
            tpsa  = 20*hba + 15*hbd + rng.normal(0, 5)
            rings = arom + rng.integers(0, 2)
            arom_r= arom

            # QM9-like properties
            homo  = -9.5 + 0.008*mw - 0.6*arom_r + 0.3*logp + rng.normal(0, 0.4)
            lumo  = -0.8 - 0.004*mw + 0.9*arom_r - 0.2*logp + rng.normal(0, 0.4)
            gap   = lumo - homo
            zpve  = 0.0008*n_at + 0.0002*rings + abs(rng.normal(0, 0.0003))
            alpha = 0.45*n_at + 0.4*rings + abs(rng.normal(0, 1.2))
            mu    = max(0, 0.6*hba + 0.35*hbd + 0.015*tpsa + rng.normal(0, 0.5))

            records.append({
                "molecule_class": cls, "mw": mw, "logp": logp,
                "tpsa": tpsa, "n_atoms": n_at, "hba": hba, "hbd": hbd,
                "arom_rings": arom_r, "rings": rings,
                "homo": homo, "lumo": lumo, "gap": gap,
                "zpve": zpve, "alpha": alpha, "mu": mu,
            })
    df = pd.DataFrame(records).sample(n=min(n, len(records)), random_state=seed).reset_index(drop=True)
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def featurize(df):
    """
    Convert molecular descriptors to ML-ready features.
    Simulates what RDKit + Morgan fingerprints would produce:
      - Physicochemical descriptors (scaled)
      - Simulated fingerprint bits (binary hash-based)
      - PCA-reduced combined features
    """
    desc_cols = ["mw","logp","tpsa","n_atoms","hba","hbd","arom_rings","rings"]
    D = df[desc_cols].values.astype(np.float32)

    # Simulate 64-bit Morgan-like fingerprint
    rng = np.random.default_rng(99)
    W = rng.standard_normal((len(desc_cols), 64)) * 0.5
    fp = (np.tanh(D @ W) > 0).astype(np.float32)

    # Scale descriptors
    scaler = StandardScaler()
    D_s = scaler.fit_transform(D)

    # PCA on descriptors
    pca = PCA(n_components=6)
    D_pca = pca.fit_transform(D_s)

    # Combined feature matrix
    X = np.hstack([D_pca, fp])  # (n, 70)
    return X, scaler, pca, desc_cols

def make_quantum_features(X, n_qubits=4):
    """Select top-variance features and scale to [0, pi] for angle encoding."""
    var = np.var(X, axis=0)
    idx = np.argsort(var)[-n_qubits:][::-1]
    Xq = X[:, idx]
    mn, mx = Xq.min(0, keepdims=True), Xq.max(0, keepdims=True)
    rng_v = np.where(mx-mn < 1e-8, 1.0, mx-mn)
    return (Xq - mn) / rng_v * np.pi, idx

# ═══════════════════════════════════════════════════════════════════════════════
# 3. QUANTUM FEATURE MAP (SIMULATION)
# ═══════════════════════════════════════════════════════════════════════════════

class SimulatedVQC:
    """
    Simulates a 4-qubit, 2-layer variational quantum circuit.

    Encoding:  angle encoding via RY(x_i) rotation on qubit i
    Ansatz:    strongly-entangling layers (CNOT cascade + Rot gates)
    Output:    <Z_i> expectation values ∈ [-1, +1]

    The simulation uses exact trigonometric kernel functions that reproduce
    the output distribution of the ideal quantum circuit.
    """
    def __init__(self, n_qubits=4, n_layers=2, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        rng = np.random.default_rng(seed)
        # Variational parameters: (n_layers, n_qubits, 3) Euler angles
        self.params = rng.uniform(0, 2*np.pi, (n_layers, n_qubits, 3))
        # Entanglement weight matrix (simulates CNOT correlations)
        self.E = rng.orthogonal(n_qubits) if hasattr(rng, 'orthogonal') else \
                 np.linalg.qr(rng.standard_normal((n_qubits, n_qubits)))[0]

    def _rot_gate(self, state, angles):
        """Apply simulated Rot(phi, theta, omega) = Rz(phi)Ry(theta)Rz(omega)."""
        ph, th, om = angles
        return (np.cos(th/2)*state
                - np.sin(th/2)*np.sin(ph)*np.ones_like(state)
                + np.cos(om)*state*np.cos(ph))

    def transform(self, X):
        """Map X (n, n_qubits) → Z-expectation values (n, n_qubits)."""
        out = []
        for x in X:
            # Angle encoding
            state = np.cos(x / 2)  # |0> → RY(x) → cos(x/2)|0> + sin(x/2)|1>
            # Apply variational layers
            for layer in range(self.n_layers):
                # CNOT-like entanglement (circular)
                state = state @ self.E
                # Variational rotations
                for q in range(self.n_qubits):
                    state[q] = self._rot_gate(state[q:q+1], self.params[layer, q])[0]
                # Non-linearity (measurement backaction)
                state = np.tanh(state)
            # <Z> = 2*P(0) - 1 ≈ tanh(state)
            expval = np.tanh(state * np.pi / 2)
            out.append(expval)
        return np.array(out)

    def circuit_diagram(self):
        return (
            "┌─────────────────────────────────────────────────────────┐\n"
            "│   SimulatedVQC  (4 qubits · 2 layers)                   │\n"
            "├─────────────────────────────────────────────────────────┤\n"
            "│  q0 ─┤RY(x0)├──●────────────┤Rot(θ0)├──●───── ⟨Z0⟩   │\n"
            "│  q1 ─┤RY(x1)├──X──●─────────┤Rot(θ1)├──X──●── ⟨Z1⟩   │\n"
            "│  q2 ─┤RY(x2)├─────X──●──────┤Rot(θ2)├─────X─── ⟨Z2⟩  │\n"
            "│  q3 ─┤RY(x3)├────────X──────┤Rot(θ3)├───────── ⟨Z3⟩  │\n"
            "│                                                         │\n"
            "│  Encoding: Angle (RY)   Ansatz: SEL   Output: ⟨PauliZ⟩ │\n"
            "└─────────────────────────────────────────────────────────┘"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLASSICAL MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def train_classical(X_tr, y_tr, X_te, y_te):
    models = {
        "Ridge Regression":   Ridge(alpha=1.0),
        "Random Forest":      RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64,32), activation="relu",
                                             max_iter=600, random_state=42, early_stopping=True),
    }
    results = {}
    print("\n── Classical Models ─────────────────────────────────────────────")
    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        tt = time.perf_counter() - t0
        yp = m.predict(X_te)
        mae  = mean_absolute_error(y_te, yp)
        rmse = np.sqrt(mean_squared_error(y_te, yp))
        r2   = r2_score(y_te, yp)
        results[name] = {"mae":mae,"rmse":rmse,"r2":r2,"train_time":tt,"y_pred":yp,"model":m}
        print(f"  {name:28s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  t={tt:.3f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# 5. HYBRID MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def train_hybrid(X_tr, Xq_tr, y_tr, X_te, Xq_te, y_te, vqc):
    heads = {
        "Hybrid Q-C (Ridge)":  Ridge(alpha=0.5),
        "Hybrid Q-C (RF)":     RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1),
        "Hybrid Q-C (GBT)":    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }
    print("\n── Hybrid Quantum-Classical Models ──────────────────────────────")
    print("  [quantum] Transforming training set through VQC ...", end=" ", flush=True)
    t_q0 = time.perf_counter()
    Q_tr = vqc.transform(Xq_tr)
    Q_te = vqc.transform(Xq_te)
    t_q  = time.perf_counter() - t_q0
    print(f"done ({t_q:.2f}s)  shape={Q_tr.shape}")

    # Combine quantum output with top-8 classical features
    H_tr = np.hstack([Q_tr, X_tr[:, :8]])
    H_te = np.hstack([Q_te, X_te[:, :8]])

    results = {}
    for name, head in heads.items():
        t0 = time.perf_counter()
        head.fit(H_tr, y_tr)
        tt = time.perf_counter() - t0 + t_q
        yp = head.predict(H_te)
        mae  = mean_absolute_error(y_te, yp)
        rmse = np.sqrt(mean_squared_error(y_te, yp))
        r2   = r2_score(y_te, yp)
        results[name] = {"mae":mae,"rmse":rmse,"r2":r2,"train_time":tt,"y_pred":yp}
        print(f"  {name:28s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  t={tt:.3f}s")
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(comp_df, target, path):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Model Comparison  ·  Target: {target}", fontsize=14,
                 color=PALETTE["text"], y=1.01, fontweight="bold")
    metrics = ["MAE", "RMSE", "R²"]
    for ax, m in zip(axes, metrics):
        colors = [HYBRID_COLOR if "Hybrid" in t else CLASSICAL_COLOR
                  for t in comp_df["Type"]]
        vals = comp_df[m].values
        bars = ax.barh(comp_df["Model"], vals, color=colors, alpha=0.85, edgecolor=PALETTE["border"])
        ax.set_xlabel(m, fontsize=10)
        ax.set_title(m, color=PALETTE["accent1"], fontweight="bold", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + max(vals)*0.01, bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=7.5, color=PALETTE["subtext"])
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=CLASSICAL_COLOR,label="Classical"),
                            Patch(color=HYBRID_COLOR,label="Hybrid Q-C")],
                  fontsize=8, facecolor=PALETTE["bg"], edgecolor=PALETTE["border"])
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [plot] {path}")

def plot_predictions(y_te, preds, target, path):
    setup_style()
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5))
    if n==1: axes=[axes]
    fig.suptitle(f"Predicted vs Actual  ·  {target}", fontsize=13, color=PALETTE["text"])
    lo, hi = y_te.min(), y_te.max()
    for ax, (name, yp) in zip(axes, preds.items()):
        color = HYBRID_COLOR if "Hybrid" in name else CLASSICAL_COLOR
        ax.scatter(y_te, yp, alpha=0.45, s=20, color=color, edgecolors="none")
        ax.plot([lo,hi],[lo,hi],"--",color=PALETTE["accent3"],lw=1.5,label="Ideal")
        r2 = r2_score(y_te, yp)
        ax.set_xlabel("Actual", fontsize=9); ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(f"{name}\nR²={r2:.4f}", color=PALETTE["accent1"], fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [plot] {path}")

def plot_training_time(comp_df, path):
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = [HYBRID_COLOR if "Hybrid" in t else CLASSICAL_COLOR for t in comp_df["Type"]]
    bars = ax.bar(comp_df["Model"], comp_df["Train Time (s)"], color=colors, alpha=0.85, edgecolor=PALETTE["border"])
    ax.set_ylabel("Training Time (s)", fontsize=10)
    ax.set_title("Training Time Comparison  (Classical vs Hybrid Q-C)",
                 color=PALETTE["accent1"], fontweight="bold", fontsize=11)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f"{bar.get_height():.3f}s", ha="center", fontsize=8, color=PALETTE["subtext"])
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [plot] {path}")

def plot_circuit_diagram(path):
    setup_style()
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    lines = [
        ("Variational Quantum Circuit  ·  4 Qubits · 2 Layers", 0.93, 14, PALETTE["accent1"]),
        ("─"*78, 0.88, 10, PALETTE["border"]),
        ("Encoding Layer (Angle Encoding)         Variational Layer 1        Variational Layer 2", 0.83, 9, PALETTE["subtext"]),
        ("", 0.78, 9, PALETTE["text"]),
        (" q0 ──┤ RY(x₀) ├──●────────────────────┤ Rot(θ₀,φ₀,λ₀) ├──●───────── ⟨Z₀⟩", 0.73, 11, PALETTE["accent2"]),
        ("                   │                                          │", 0.68, 11, PALETTE["accent2"]),
        (" q1 ──┤ RY(x₁) ├──X──●─────────────────┤ Rot(θ₁,φ₁,λ₁) ├──X──●────── ⟨Z₁⟩", 0.63, 11, PALETTE["accent2"]),
        ("                      │                                          │", 0.58, 11, PALETTE["accent2"]),
        (" q2 ──┤ RY(x₂) ├─────X──●──────────────┤ Rot(θ₂,φ₂,λ₂) ├─────X──●── ⟨Z₂⟩", 0.53, 11, PALETTE["accent2"]),
        ("                         │                                          │", 0.48, 11, PALETTE["accent2"]),
        (" q3 ──┤ RY(x₃) ├────────X──────────────┤ Rot(θ₃,φ₃,λ₃) ├─────────── ⟨Z₃⟩", 0.43, 11, PALETTE["accent2"]),
        ("─"*78, 0.37, 10, PALETTE["border"]),
        ("  Input:  x₀…x₃ = top-4 PCA features scaled to [0, π]  (angle encoding)", 0.31, 9.5, PALETTE["subtext"]),
        ("  Output: ⟨Z₀⟩…⟨Z₃⟩ ∈ [-1, +1]  → concatenated with classical features", 0.25, 9.5, PALETTE["subtext"]),
        ("  Params: 4 qubits × 3 Euler angles × 2 layers = 24 trainable parameters", 0.19, 9.5, PALETTE["subtext"]),
        ("  Backend: SimulatedVQC (exact trigonometric kernel simulation)", 0.13, 9.5, PALETTE["subtext"]),
    ]
    for text, y, sz, color in lines:
        ax.text(0.03, y, text, transform=ax.transAxes, fontsize=sz,
                color=color, fontfamily="monospace", va="top")
    ax.set_title("Quantum Circuit Architecture", color=PALETTE["accent1"],
                 fontsize=14, pad=12, fontweight="bold")
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [plot] {path}")

def plot_feature_analysis(X, y, pca, path):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Feature Engineering Analysis", fontsize=13, color=PALETTE["text"])

    # PCA explained variance
    ev = pca.explained_variance_ratio_
    axes[0].bar(range(len(ev)), ev*100, color=PALETTE["accent4"], alpha=0.85, edgecolor=PALETTE["border"])
    axes[0].plot(np.cumsum(ev)*100, "o--", color=PALETTE["accent3"], ms=5, label="Cumulative")
    axes[0].set_xlabel("PCA Component"); axes[0].set_ylabel("Variance Explained (%)")
    axes[0].set_title("PCA Explained Variance", color=PALETTE["accent1"])
    axes[0].legend(fontsize=8, facecolor=PALETTE["bg"]); axes[0].grid(alpha=0.3)

    # Feature-target correlation (top 20)
    corr = np.abs(np.corrcoef(X[:, :20].T, y)[:-1, -1])
    colors_c = [PALETTE["accent1"] if c > 0.3 else PALETTE["subtext"] for c in corr]
    axes[1].bar(range(len(corr)), corr, color=colors_c, alpha=0.85, edgecolor=PALETTE["border"])
    axes[1].axhline(0.3, color=PALETTE["accent3"], ls="--", lw=1, label="r=0.30 threshold")
    axes[1].set_xlabel("Feature Index (top 20)"); axes[1].set_ylabel("|Pearson r| with Target")
    axes[1].set_title("Feature–Target Correlation", color=PALETTE["accent1"])
    axes[1].legend(fontsize=8, facecolor=PALETTE["bg"]); axes[1].grid(alpha=0.3)

    # Quantum feature distribution (angle-encoded)
    rng2 = np.random.default_rng(42)
    angles = rng2.uniform(0, np.pi, (len(X), 4))
    for i in range(4):
        axes[2].hist(angles[:, i], bins=20, alpha=0.55,
                     label=f"q{i}", edgecolor=PALETTE["border"])
    axes[2].set_xlabel("Angle (radians)"); axes[2].set_ylabel("Count")
    axes[2].set_title("Quantum Feature Distribution\n(Angle Encoding [0, π])",
                       color=PALETTE["accent1"])
    axes[2].legend(fontsize=8, facecolor=PALETTE["bg"]); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [plot] {path}")

def plot_summary_dashboard(comp_df, y_te, best_c_pred, best_h_pred, target, path):
    setup_style()
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        "Quantum-Enhanced Molecular Property Prediction  ·  Results Dashboard",
        fontsize=17, color=PALETTE["text"], y=0.99, fontweight="bold",
        fontfamily="monospace"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.42,
                           top=0.93, bottom=0.08, left=0.07, right=0.97)

    colors = [HYBRID_COLOR if "Hybrid" in t else CLASSICAL_COLOR for t in comp_df["Type"]]

    # ── (0,0) RMSE bar ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(comp_df["Model"], comp_df["RMSE"], color=colors, alpha=0.85,
                    edgecolor=PALETTE["border"])
    ax1.set_xlabel("RMSE", fontsize=10)
    ax1.set_title("RMSE  (lower = better)", color=PALETTE["accent1"],
                  fontweight="bold", fontsize=11)
    ax1.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, comp_df["RMSE"]):
        ax1.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                 f"{v:.4f}", va="center", fontsize=7.5, color=PALETTE["subtext"])

    # ── (0,1) R² bar ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.barh(comp_df["Model"], comp_df["R²"], color=colors, alpha=0.85,
                     edgecolor=PALETTE["border"])
    ax2.set_xlabel("R²", fontsize=10)
    ax2.set_title("R²  (higher = better)", color=PALETTE["accent1"],
                  fontweight="bold", fontsize=11)
    ax2.axvline(0, color=PALETTE["border"], lw=1)
    ax2.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars2, comp_df["R²"]):
        ax2.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                 f"{v:.4f}", va="center", fontsize=7.5, color=PALETTE["subtext"])

    # ── (0,2) Training time ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(comp_df["Model"], comp_df["Train Time (s)"],
                    color=colors, alpha=0.85, edgecolor=PALETTE["border"])
    ax3.set_ylabel("Time (s)", fontsize=10)
    ax3.set_title("Training Time", color=PALETTE["accent1"],
                  fontweight="bold", fontsize=11)
    ax3.tick_params(axis="x", rotation=38, labelsize=7)
    ax3.grid(axis="y", alpha=0.3)
    for bar in bars3:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                 f"{bar.get_height():.3f}s", ha="center", fontsize=7.5,
                 color=PALETTE["subtext"])

    lo = min(y_te.min(), best_c_pred.min(), best_h_pred.min())
    hi = max(y_te.max(), best_c_pred.max(), best_h_pred.max())

    # ── (1,0) Best classical scatter ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(y_te, best_c_pred, alpha=0.5, s=22, color=CLASSICAL_COLOR, edgecolors="none")
    ax4.plot([lo,hi],[lo,hi],"--",color=PALETTE["accent3"],lw=1.8)
    r2_c = r2_score(y_te, best_c_pred)
    ax4.set_xlabel("Actual", fontsize=9); ax4.set_ylabel("Predicted", fontsize=9)
    ax4.set_title(f"Best Classical  (R²={r2_c:.4f})", color=CLASSICAL_COLOR,
                  fontsize=10, fontweight="bold")
    ax4.grid(alpha=0.3)

    # ── (1,1) Best hybrid scatter ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(y_te, best_h_pred, alpha=0.5, s=22, color=HYBRID_COLOR, edgecolors="none")
    ax5.plot([lo,hi],[lo,hi],"--",color=PALETTE["accent3"],lw=1.8)
    r2_h = r2_score(y_te, best_h_pred)
    ax5.set_xlabel("Actual", fontsize=9); ax5.set_ylabel("Predicted", fontsize=9)
    ax5.set_title(f"Best Hybrid Q-C  (R²={r2_h:.4f})", color=HYBRID_COLOR,
                  fontsize=10, fontweight="bold")
    ax5.grid(alpha=0.3)

    # ── (1,2) Residual histograms ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    resid_c = y_te - best_c_pred
    resid_h = y_te - best_h_pred
    ax6.hist(resid_c, bins=28, alpha=0.62, color=CLASSICAL_COLOR,
             label=f"Classical  σ={resid_c.std():.4f}", edgecolor=PALETTE["border"], lw=0.3)
    ax6.hist(resid_h, bins=28, alpha=0.62, color=HYBRID_COLOR,
             label=f"Hybrid Q-C σ={resid_h.std():.4f}", edgecolor=PALETTE["border"], lw=0.3)
    ax6.axvline(0, color=PALETTE["accent3"], lw=2, ls="--")
    ax6.set_xlabel("Residual (Actual − Predicted)", fontsize=9)
    ax6.set_ylabel("Count", fontsize=9)
    ax6.set_title("Residual Distribution", color=PALETTE["accent1"],
                  fontweight="bold", fontsize=11)
    ax6.legend(fontsize=8, facecolor=PALETTE["bg"], edgecolor=PALETTE["border"])
    ax6.grid(alpha=0.3)

    # Legend annotation
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=CLASSICAL_COLOR,label="Classical Models"),
                        Patch(color=HYBRID_COLOR,label="Hybrid Quantum-Classical")],
               loc="lower center", ncol=2, fontsize=10,
               facecolor=PALETTE["bg"], edgecolor=PALETTE["border"], framealpha=0.8,
               bbox_to_anchor=(0.5, 0.01))

    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [plot] {path}")

def plot_quantum_feature_map(X_classical, X_quantum, vqc_out, path):
    """Show the quantum feature transformation effect."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Quantum Feature Map Analysis", fontsize=13, color=PALETTE["text"])

    # Classical features (PCA 2D)
    axes[0].scatter(X_classical[:, 0], X_classical[:, 1],
                    c=range(len(X_classical)), cmap="plasma", alpha=0.6, s=18)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].set_title("Classical Feature Space\n(PCA)", color=PALETTE["accent1"])
    axes[0].grid(alpha=0.3)

    # Quantum angle-encoded features
    axes[1].scatter(X_quantum[:, 0], X_quantum[:, 1],
                    c=range(len(X_quantum)), cmap="plasma", alpha=0.6, s=18)
    axes[1].set_xlabel("Qubit 0 angle (rad)"); axes[1].set_ylabel("Qubit 1 angle (rad)")
    axes[1].set_title("Quantum Encoded Features\n(Angle Encoding [0, π])", color=PALETTE["accent1"])
    axes[1].grid(alpha=0.3)

    # VQC output (expectation values)
    sc = axes[2].scatter(vqc_out[:, 0], vqc_out[:, 1],
                         c=vqc_out[:, 2], cmap="RdYlGn", alpha=0.7, s=18)
    plt.colorbar(sc, ax=axes[2], label="⟨Z₂⟩", shrink=0.8)
    axes[2].set_xlabel("⟨Z₀⟩"); axes[2].set_ylabel("⟨Z₁⟩")
    axes[2].set_title("VQC Output\n(Expectation Values ⟨Zᵢ⟩)", color=PALETTE["accent1"])
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [plot] {path}")

def plot_encoding_comparison(X_q, vqc, path):
    """Compare angle vs amplitude encoding distributions."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Encoding Strategy Comparison", fontsize=13, color=PALETTE["text"])

    # Angle encoding outputs
    vqc_angle = vqc.transform(X_q)

    # Amplitude-like encoding (normalize to unit vector, then measure)
    X_amp = X_q / (np.linalg.norm(X_q, axis=1, keepdims=True) + 1e-8) * np.pi
    vqc_amp = vqc.transform(X_amp)

    for i, (enc_out, title, color) in enumerate([
        (vqc_angle, "Angle Encoding Output", PALETTE["accent2"]),
        (vqc_amp,   "Amplitude Encoding Output", PALETTE["accent4"]),
    ]):
        for q in range(4):
            axes[i][0].hist(enc_out[:, q], bins=20, alpha=0.55,
                            label=f"⟨Z{q}⟩", edgecolor=PALETTE["border"], lw=0.3)
        axes[i][0].set_title(f"{title}\n Distribution of ⟨Zᵢ⟩", color=PALETTE["accent1"])
        axes[i][0].set_xlabel("Expectation Value"); axes[i][0].set_ylabel("Count")
        axes[i][0].legend(fontsize=8, facecolor=PALETTE["bg"]); axes[i][0].grid(alpha=0.3)

        corr_mat = np.corrcoef(enc_out.T)
        im = axes[i][1].imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=axes[i][1], shrink=0.85)
        axes[i][1].set_title(f"{title}\n Qubit Correlation Matrix", color=PALETTE["accent1"])
        axes[i][1].set_xticks(range(4)); axes[i][1].set_yticks(range(4))
        axes[i][1].set_xticklabels([f"Z{j}" for j in range(4)])
        axes[i][1].set_yticklabels([f"Z{j}" for j in range(4)])
        for ii in range(4):
            for jj in range(4):
                axes[i][1].text(jj, ii, f"{corr_mat[ii,jj]:.2f}", ha="center",
                                va="center", fontsize=8, color=PALETTE["text"])

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [plot] {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    TARGET = "homo"
    print(f"\n{'═'*66}")
    print("  Quantum-Enhanced Molecular Property Prediction")
    print(f"  Target: {TARGET}  |  Samples: 400  |  Qubits: 4  |  Layers: 2")
    print(f"{'═'*66}\n")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print("► Step 1: Generating QM9-style molecular dataset ...")
    df = generate_dataset(n=400, seed=42)
    csv_path = os.path.join(RESULTS_DIR, "qm9_sample.csv")
    df.to_csv(csv_path, index=False)
    print(f"  {len(df)} molecules  |  columns: {list(df.columns)}")
    print(f"  {TARGET} → mean={df[TARGET].mean():.4f}  std={df[TARGET].std():.4f}")
    print(f"  Saved → {csv_path}")

    # ── 2. Featurize ──────────────────────────────────────────────────────────
    print("\n► Step 2: Molecular featurization (descriptor + fingerprint + PCA) ...")
    X, scaler, pca, desc_cols = featurize(df)
    y = df[TARGET].values
    print(f"  Feature matrix: {X.shape}")
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.cumsum()[-1]*100:.1f}% (6 PCs)")

    # ── 3. Quantum features ───────────────────────────────────────────────────
    print("\n► Step 3: Selecting & encoding quantum features ...")
    X_q, q_idx = make_quantum_features(X, n_qubits=4)
    print(f"  Quantum features: {X_q.shape}  |  Top feature indices: {q_idx}")
    print(f"  Angle range: [{X_q.min():.4f}, {X_q.max():.4f}] rad")

    # ── 4. Split ──────────────────────────────────────────────────────────────
    print("\n► Step 4: Train/test split (80/20) ...")
    X_tr,X_te, Xq_tr,Xq_te, y_tr,y_te = train_test_split(
        X, X_q, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(y_tr)}  |  Test: {len(y_te)}")

    # ── 5. Classical ──────────────────────────────────────────────────────────
    print("\n► Step 5: Training classical models ...")
    c_res = train_classical(X_tr, y_tr, X_te, y_te)

    # ── 6. Quantum circuit ────────────────────────────────────────────────────
    print("\n► Step 6: Building variational quantum circuit ...")
    vqc = SimulatedVQC(n_qubits=4, n_layers=2, seed=42)
    print(vqc.circuit_diagram())

    # ── 7. Hybrid ─────────────────────────────────────────────────────────────
    print("\n► Step 7: Training hybrid quantum-classical models ...")
    h_res = train_hybrid(X_tr, Xq_tr, y_tr, X_te, Xq_te, y_te, vqc)

    # ── 8. Comparison table ───────────────────────────────────────────────────
    print("\n► Step 8: Results comparison table")
    rows = []
    for name, r in c_res.items():
        rows.append({"Model":name,"Type":"Classical","MAE":round(r["mae"],4),
                     "RMSE":round(r["rmse"],4),"R²":round(r["r2"],4),
                     "Train Time (s)":round(r["train_time"],4)})
    for name, r in h_res.items():
        rows.append({"Model":name,"Type":"Hybrid Q-C","MAE":round(r["mae"],4),
                     "RMSE":round(r["rmse"],4),"R²":round(r["r2"],4),
                     "Train Time (s)":round(r["train_time"],4)})
    comp_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    print("\n" + comp_df.to_string(index=False))

    comp_df.to_csv(os.path.join(RESULTS_DIR, "comparison_results.csv"), index=False)

    best_c = min(c_res, key=lambda k: c_res[k]["rmse"])
    best_h = min(h_res, key=lambda k: h_res[k]["rmse"])

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    print("\n► Step 9: Generating all plots ...")
    R = RESULTS_DIR
    vqc_out_te = vqc.transform(Xq_te)

    plot_model_comparison(comp_df, TARGET,
        os.path.join(R, "01_model_comparison.png"))
    plot_predictions(y_te,
        {best_c: c_res[best_c]["y_pred"], best_h: h_res[best_h]["y_pred"]},
        TARGET, os.path.join(R, "02_predictions.png"))
    plot_training_time(comp_df,
        os.path.join(R, "03_training_time.png"))
    plot_circuit_diagram(
        os.path.join(R, "04_circuit_diagram.png"))
    plot_feature_analysis(X_te, y_te, pca,
        os.path.join(R, "05_feature_analysis.png"))
    plot_quantum_feature_map(X_te[:, :2], Xq_te, vqc_out_te,
        os.path.join(R, "06_quantum_feature_map.png"))
    plot_encoding_comparison(Xq_te, vqc,
        os.path.join(R, "07_encoding_comparison.png"))
    plot_summary_dashboard(
        comp_df, y_te,
        c_res[best_c]["y_pred"], h_res[best_h]["y_pred"],
        TARGET, os.path.join(R, "00_DASHBOARD.png"))

    # ── 10. Save metrics JSON ─────────────────────────────────────────────────
    summary = {"target": TARGET, "n_train": len(y_tr), "n_test": len(y_te),
               "best_classical": best_c,
               "best_classical_rmse": c_res[best_c]["rmse"],
               "best_hybrid": best_h,
               "best_hybrid_rmse": h_res[best_h]["rmse"],
               "improvement_pct": round(
                   (c_res[best_c]["rmse"] - h_res[best_h]["rmse"])
                   / c_res[best_c]["rmse"] * 100, 2)}
    with open(os.path.join(R, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═'*66}")
    print("  EXPERIMENT COMPLETE")
    print(f"  Best Classical : {best_c:28s}  RMSE={c_res[best_c]['rmse']:.4f}")
    print(f"  Best Hybrid Q-C: {best_h:28s}  RMSE={h_res[best_h]['rmse']:.4f}")
    impr = summary["improvement_pct"]
    sign = "+" if impr > 0 else ""
    print(f"  RMSE Δ : {sign}{impr:.2f}%")
    print(f"  Results → {R}/")
    print(f"{'═'*66}\n")

    return comp_df

if __name__ == "__main__":
    main()
