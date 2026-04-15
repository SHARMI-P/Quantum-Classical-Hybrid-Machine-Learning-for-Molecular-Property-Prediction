import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

HARTREE_TO_EV = 27.2114
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)

# ════════════════════════════════════════════════
# STEP 1 — Load real QM9 features
# ════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading real QM9 data")
print("=" * 60)

df = pd.read_csv('data/qm9_features.csv')
feature_cols = [c for c in df.columns if c not in ['homo_energy', 'smiles']]

X = df[feature_cols].values.astype(np.float64)
y = df['homo_energy'].values * HARTREE_TO_EV

print(f"Molecules  : {len(X)}")
print(f"Features   : {len(feature_cols)}")
print(f"HOMO range : {y.min():.3f} to {y.max():.3f} eV")
print(f"HOMO mean  : {y.mean():.3f} eV  std: {y.std():.3f} eV")

# ════════════════════════════════════════════════
# STEP 2 — Proper 3-way split
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Creating proper 3-way split")
print("=" * 60)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.176, random_state=RANDOM_SEED
)

# Scale features
feat_scaler = StandardScaler()
X_train_s   = feat_scaler.fit_transform(X_train)
X_val_s     = feat_scaler.transform(X_val)
X_test_s    = feat_scaler.transform(X_test)

# FIX: Also normalise y so VQC loss is on a sane scale (~0 to 1)
y_mean  = y_train.mean()
y_std   = y_train.std()
y_train_n = (y_train - y_mean) / y_std
y_val_n   = (y_val   - y_mean) / y_std
y_test_n  = (y_test  - y_mean) / y_std

print(f"Train size      : {len(X_train)}")
print(f"Validation size : {len(X_val)}")
print(f"Test size       : {len(X_test)}  (held out)")
print(f"y normalised    : mean={y_mean:.3f} eV, std={y_std:.3f} eV")
print(f"  (predictions will be un-normalised for RMSE reporting)")

# ════════════════════════════════════════════════
# STEP 3 — PennyLane VQC with Adam optimizer
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Training VQC with PennyLane + Adam optimizer")
print("=" * 60)

N_QUBITS = 6
N_LAYERS = 2
N_STEPS  = 300   # More steps for better convergence
LR       = 0.03  # Slightly lower LR for stability
BATCH    = 80

dev = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="autograd")
def vqc_circuit(params, x):
    # Angle encoding of first 6 (normalised) features
    for i in range(N_QUBITS):
        qml.RY(x[i] * np.pi, wires=i)
    # Variational layers
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(params[l, i, 0], wires=i)
            qml.RZ(params[l, i, 1], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

def cost_fn_train(params, X_batch, y_batch_n):
    """
    Autograd-compatible cost.
    Uses normalised y so loss is on a ~0-1 scale.
    Linear regression head (least squares) on quantum features.
    """
    q_feats = []
    for x in X_batch:
        x_enc = pnp.array(x[:N_QUBITS], requires_grad=False)
        out   = vqc_circuit(params, x_enc)
        q_feats.append(pnp.stack(out))
    Q = pnp.stack(q_feats)                          # shape (batch, N_QUBITS)

    # Closed-form linear head: w = (Q^T Q)^-1 Q^T y
    # Use autograd-compatible operations
    y_t = pnp.array(y_batch_n, requires_grad=False)
    QtQ = pnp.dot(Q.T, Q) + 1e-4 * pnp.eye(N_QUBITS)  # ridge term for stability
    Qty = pnp.dot(Q.T, y_t)
    w   = pnp.dot(pnp.linalg.inv(QtQ), Qty)
    y_pred = pnp.dot(Q, w)
    return pnp.mean((y_pred - y_t) ** 2)

def get_quantum_features(params, X_data):
    """Extract quantum feature vectors — plain numpy, not inside autograd."""
    np_params = pnp.array(params, requires_grad=False)
    out = []
    for x in X_data:
        x_enc = pnp.array(x[:N_QUBITS], requires_grad=False)
        q_out = vqc_circuit(np_params, x_enc)
        out.append([float(v) for v in q_out])
    return np.array(out)

# Initialise parameters
params = pnp.array(
    np.random.uniform(-np.pi, np.pi, (N_LAYERS, N_QUBITS, 2)),
    requires_grad=True
)

opt = qml.AdamOptimizer(stepsize=LR)

print(f"Circuit   : {N_QUBITS} qubits x {N_LAYERS} layers = "
      f"{N_LAYERS * N_QUBITS * 2} parameters")
print(f"Optimizer : Adam (lr={LR}), {N_STEPS} steps, batch={BATCH}")
print(f"Target    : normalised HOMO energy (should see loss ~0.5-1.0 now)")
print(f"Training on {len(X_train)} molecules...\n")

loss_history  = []
best_val_loss = np.inf
best_params   = params.copy()

for step in range(N_STEPS):
    idx  = np.random.choice(len(X_train_s), BATCH, replace=False)
    X_b  = X_train_s[idx]
    y_b  = y_train_n[idx]      # use normalised y for VQC training

    params, loss = opt.step_and_cost(
        lambda p: cost_fn_train(p, X_b, y_b), params
    )
    loss_history.append(float(loss))

    if (step + 1) % 30 == 0:
        val_loss = float(cost_fn_train(
            pnp.array(params, requires_grad=False),
            X_val_s[:40], y_val_n[:40]
        ))
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_params   = pnp.array(params, requires_grad=False).copy()
        marker = " <- best" if is_best else ""
        print(f"  Step {step+1:3d}/{N_STEPS} | "
              f"train_loss={loss:.4f} | val_loss={val_loss:.4f}{marker}")

print(f"\nBest validation loss : {best_val_loss:.4f}  "
      f"(should be << 1.0 if VQC is learning)")
print(f"Loss reduction       : {loss_history[0]:.4f} -> "
      f"{loss_history[-1]:.4f} "
      f"({(1 - loss_history[-1]/loss_history[0])*100:.1f}% reduction)")

# ════════════════════════════════════════════════
# STEP 4 — Extract quantum features
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Extracting quantum features using best VQC params")
print("=" * 60)

print("Processing train set ...")
X_train_q = get_quantum_features(best_params, X_train_s)
print("Processing validation set ...")
X_val_q   = get_quantum_features(best_params, X_val_s)
print("Processing test set ...")
X_test_q  = get_quantum_features(best_params, X_test_s)

# Hybrid = classical features + quantum features
X_train_hq = np.hstack([X_train_s, X_train_q])
X_val_hq   = np.hstack([X_val_s,   X_val_q])
X_test_hq  = np.hstack([X_test_s,  X_test_q])

print(f"Classical features : {X_train_s.shape[1]}")
print(f"Quantum features   : {X_train_q.shape[1]}")
print(f"Hybrid features    : {X_train_hq.shape[1]}")

# ════════════════════════════════════════════════
# STEP 5 — Tune all models on raw eV targets
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Tuning all models with GridSearchCV")
print("=" * 60)

def tune_and_eval(name, estimator, param_grid, X_tr, y_tr, X_te, y_te):
    gs = GridSearchCV(estimator, param_grid, cv=5,
                      scoring='neg_root_mean_squared_error', n_jobs=-1)
    gs.fit(X_tr, y_tr)
    y_pred = gs.best_estimator_.predict(X_te)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    r2     = r2_score(y_te, y_pred)
    print(f"  {name:28s} | best={gs.best_params_} | "
          f"RMSE={rmse:.4f} eV | R2={r2:.4f}")
    return rmse, r2, gs.best_estimator_

results = {}

# Train all models on raw eV (not normalised) for fair comparison
print("\nClassical models (properly tuned):")
results['Ridge'] = tune_and_eval(
    'Ridge (tuned)', Ridge(),
    {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
    X_train_s, y_train, X_test_s, y_test
)
results['ElasticNet'] = tune_and_eval(
    'ElasticNet (tuned)', ElasticNet(max_iter=10000),
    {'alpha': [0.001, 0.01, 0.1], 'l1_ratio': [0.3, 0.5, 0.7]},
    X_train_s, y_train, X_test_s, y_test
)
results['SVR'] = tune_and_eval(
    'SVR (tuned)', SVR(),
    {'C': [1.0, 10.0, 100.0], 'gamma': ['scale', 'auto'], 'epsilon': [0.01, 0.1]},
    X_train_s, y_train, X_test_s, y_test
)
results['Random Forest'] = tune_and_eval(
    'Random Forest (tuned)', RandomForestRegressor(random_state=RANDOM_SEED),
    {'n_estimators': [100, 200], 'max_depth': [5, 10, None]},
    X_train_s, y_train, X_test_s, y_test
)

print("\nHybrid quantum-classical models:")
results['Hybrid Ridge'] = tune_and_eval(
    'Hybrid Ridge', Ridge(),
    {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
    X_train_hq, y_train, X_test_hq, y_test
)
results['Hybrid SVR'] = tune_and_eval(
    'Hybrid SVR', SVR(),
    {'C': [1.0, 10.0, 100.0], 'gamma': ['scale', 'auto'], 'epsilon': [0.01, 0.1]},
    X_train_hq, y_train, X_test_hq, y_test
)
results['Hybrid RF'] = tune_and_eval(
    'Hybrid Random Forest', RandomForestRegressor(random_state=RANDOM_SEED),
    {'n_estimators': [100, 200], 'max_depth': [5, 10, None]},
    X_train_hq, y_train, X_test_hq, y_test
)

# ════════════════════════════════════════════════
# STEP 6 — Repeated 5x2 CV: best classical vs best hybrid
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Repeated 5x2 CV  (best classical SVR vs Hybrid SVR)")
print("=" * 60)

X_cv  = np.vstack([X_train_s, X_val_s])
y_cv  = np.concatenate([y_train, y_val])
print("Extracting quantum features for full CV set...")
X_cv_hq = np.hstack([X_cv, get_quantum_features(best_params, X_cv)])

rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_SEED)

classical_rmses = []
hybrid_rmses    = []

# Use the best classical model type (SVR) for fair CV comparison
best_C     = results['SVR'][2].C
best_gamma = results['SVR'][2].gamma

print(f"Using SVR(C={best_C}, gamma={best_gamma}) for both classical and hybrid\n")

for fold, (tr, te) in enumerate(rkf.split(X_cv)):
    m_c = SVR(C=best_C, gamma=best_gamma, epsilon=0.1)
    m_c.fit(X_cv[tr], y_cv[tr])
    c_rmse = np.sqrt(mean_squared_error(y_cv[te], m_c.predict(X_cv[te])))
    classical_rmses.append(c_rmse)

    m_h = SVR(C=best_C, gamma=best_gamma, epsilon=0.1)
    m_h.fit(X_cv_hq[tr], y_cv[tr])
    h_rmse = np.sqrt(mean_squared_error(y_cv[te], m_h.predict(X_cv_hq[te])))
    hybrid_rmses.append(h_rmse)

    print(f"  Fold {fold+1:2d} | Classical={c_rmse:.4f} | Hybrid={h_rmse:.4f} | "
          f"diff={c_rmse - h_rmse:+.4f}")

t_stat, p_val  = stats.ttest_rel(classical_rmses, hybrid_rmses)
w_stat, p_wilc = stats.wilcoxon(classical_rmses, hybrid_rmses)
diff  = np.array(classical_rmses) - np.array(hybrid_rmses)
cohen = np.mean(diff) / np.std(diff)

print(f"\nClassical SVR RMSE : {np.mean(classical_rmses):.4f} +/- "
      f"{np.std(classical_rmses):.4f} eV")
print(f"Hybrid SVR RMSE    : {np.mean(hybrid_rmses):.4f} +/- "
      f"{np.std(hybrid_rmses):.4f} eV")
print(f"Mean improvement   : {np.mean(diff)*1000:.2f} meV  "
      f"({np.mean(diff)/np.mean(classical_rmses)*100:.2f}%)")
print(f"Paired t-test      : t={t_stat:.4f}, p={p_val:.4f} "
      f"({'SIGNIFICANT' if p_val < 0.05 else 'not significant'} at a=0.05)")
print(f"Wilcoxon           : W={w_stat:.1f}, p={p_wilc:.4f}")
print(f"Cohen's d          : {cohen:.4f}")

# ════════════════════════════════════════════════
# STEP 7 — Save all plots
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Saving plots")
print("=" * 60)

# Plot 1: VQC training loss curve
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='steelblue', linewidth=1.5, label='Train loss')
plt.axhline(best_val_loss, color='coral', linestyle='--',
            linewidth=1.2, label=f'Best val loss = {best_val_loss:.4f}')
plt.xlabel('Adam step')
plt.ylabel('MSE loss (normalised y)')
plt.title('VQC Training Loss — Adam Optimizer')
plt.legend()
plt.tight_layout()
plt.savefig('results/01_vqc_training_loss.png', dpi=150)
plt.close()
print("  Saved: results/01_vqc_training_loss.png")

# Plot 2: CV bar chart classical vs hybrid
labels = ['Classical SVR', 'Hybrid SVR\n(+VQC features)']
means  = [np.mean(classical_rmses), np.mean(hybrid_rmses)]
stds   = [np.std(classical_rmses),  np.std(hybrid_rmses)]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, means, yerr=stds, capsize=10,
              color=['#4C72B0', '#DD8452'], alpha=0.85, edgecolor='white',
              error_kw={'linewidth': 2})
ax.set_ylabel('RMSE (eV)')
ax.set_title(f'Classical vs Hybrid Q-C  (10-fold Repeated CV)\n'
             f'Real QM9 data | p={p_val:.4f} | Cohen\'s d={cohen:.3f}')
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.003,
            f'{m:.4f} eV', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('results/02_cv_comparison.png', dpi=150)
plt.close()
print("  Saved: results/02_cv_comparison.png")

# Plot 3: HOMO distribution
plt.figure(figsize=(7, 4))
plt.hist(y, bins=50, color='teal', edgecolor='white', alpha=0.85)
plt.xlabel('HOMO Energy (eV)')
plt.ylabel('Count')
plt.title('Real QM9 HOMO Energy Distribution (DFT-computed)\n'
          f'n=2000, mean={y.mean():.2f} eV, std={y.std():.2f} eV')
plt.tight_layout()
plt.savefig('results/03_homo_distribution.png', dpi=150)
plt.close()
print("  Saved: results/03_homo_distribution.png")

# Plot 4: All models comparison
model_names = list(results.keys())
model_rmses = [results[k][0] for k in model_names]
model_r2s   = [results[k][1] for k in model_names]
colors_bar  = ['#4C72B0', '#4C72B0', '#4C72B0', '#4C72B0',
               '#DD8452', '#DD8452', '#DD8452']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

bars1 = ax1.bar(model_names, model_rmses, color=colors_bar,
                alpha=0.85, edgecolor='white')
ax1.set_ylabel('RMSE (eV)  — lower is better')
ax1.set_title('Test Set RMSE\nBlue=Classical  Orange=Hybrid Q-C')
ax1.set_xticklabels(model_names, rotation=25, ha='right', fontsize=9)
for bar, r in zip(bars1, model_rmses):
    ax1.text(bar.get_x() + bar.get_width() / 2, r + 0.003,
             f'{r:.3f}', ha='center', fontsize=8)

bars2 = ax2.bar(model_names, model_r2s, color=colors_bar,
                alpha=0.85, edgecolor='white')
ax2.set_ylabel('R²  — higher is better')
ax2.set_title('Test Set R²\nBlue=Classical  Orange=Hybrid Q-C')
ax2.set_xticklabels(model_names, rotation=25, ha='right', fontsize=9)
for bar, r in zip(bars2, model_r2s):
    ax2.text(bar.get_x() + bar.get_width() / 2, r + 0.005,
             f'{r:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('results/04_all_models.png', dpi=150)
plt.close()
print("  Saved: results/04_all_models.png")

# Plot 5: Fold-by-fold comparison
folds = list(range(1, 11))
plt.figure(figsize=(10, 4))
plt.plot(folds, classical_rmses, 'o-', color='#4C72B0',
         linewidth=2, markersize=7, label='Classical SVR')
plt.plot(folds, hybrid_rmses,   's-', color='#DD8452',
         linewidth=2, markersize=7, label='Hybrid SVR (+VQC)')
plt.xlabel('CV Fold')
plt.ylabel('RMSE (eV)')
plt.title('Per-fold RMSE: Classical vs Hybrid\n'
          f'(lower = better,  mean diff = {np.mean(diff)*1000:.1f} meV)')
plt.xticks(folds)
plt.legend()
plt.tight_layout()
plt.savefig('results/05_fold_comparison.png', dpi=150)
plt.close()
print("  Saved: results/05_fold_comparison.png")

# ════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Dataset           : Real QM9 (DFT-computed HOMO energies)")
print(f"Molecules         : {len(X)}")
print(f"VQC optimizer     : Adam ({N_STEPS} steps, lr={LR})")
print(f"CV strategy       : Repeated 5x2 (10 folds total)")
print(f"")
print(f"Best classical    : SVR  RMSE={results['SVR'][0]:.4f} eV  R2={results['SVR'][1]:.4f}")
print(f"Best hybrid       : SVR  RMSE={results['Hybrid SVR'][0]:.4f} eV  R2={results['Hybrid SVR'][1]:.4f}")
best_model_name = max(results, key=lambda k: results[k][1])
print(f"Best overall R2   : {results[best_model_name][1]:.4f}  ({best_model_name})")
print(f"")
print(f"CV Classical RMSE : {np.mean(classical_rmses):.4f} +/- {np.std(classical_rmses):.4f} eV")
print(f"CV Hybrid RMSE    : {np.mean(hybrid_rmses):.4f} +/- {np.std(hybrid_rmses):.4f} eV")
print(f"Improvement       : {np.mean(diff)*1000:.2f} meV ({np.mean(diff)/np.mean(classical_rmses)*100:.2f}%)")
print(f"p-value (t-test)  : {p_val:.4f}")
print(f"Cohen's d         : {cohen:.4f}")
print(f"")
print(f"Plots saved to    : results/")
print("=" * 60)