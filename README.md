# Quantum-Classical Hybrid Machine Learning for Molecular Property Prediction

> A Comparative Study on the QM9 Dataset using Variational Quantum Circuits (VQC) + Classical ML

---

## 📋 Overview

This project implements and rigorously evaluates a **quantum-classical hybrid machine learning pipeline** to predict **HOMO energies** (Highest Occupied Molecular Orbital) of organic molecules from the QM9 dataset.

A **Variational Quantum Circuit (VQC)** built with PennyLane is used to generate quantum-derived molecular features, which are then combined with classical molecular descriptors and fed into scikit-learn regressors. The study benchmarks hybrid quantum-classical models against purely classical baselines using statistically rigorous evaluation.

---

## 🧪 Project Highlights

| Property | Detail |
|---|---|
| **Dataset** | QM9 — 2,000 molecules (DFT-computed HOMO energies) |
| **Framework** | PennyLane (VQC) + scikit-learn (Classical Models) |
| **Target Property** | HOMO Energy (eV) |
| **Best Model** | Random Forest — R² = 0.7676, RMSE = 0.3328 eV |
| **Statistical Test** | Repeated 5×2 CV with paired t-test & Wilcoxon signed-rank |
| **Date** | April 2026 |

---

## 🗂️ Project Structure

```
quantum_mol_project/
├── build_model.py               # Main pipeline — runs all 7 steps end-to-end
├── load_data.py                 # QM9 CSV loading and feature extraction
├── config.py                   # Hyperparameter constants and configuration
├── main.py                     # Entry point
├── run_experiment.py            # Experiment runner
├── run_simulation.py            # Simulation runner
├── run_standalone.py            # Standalone execution
├── simulate_and_plot.py         # Simulation + visualization
├── advanced_experiment.py       # Extended experiments
│
├── quantum/
│   ├── circuit.py               # PennyLane VQC circuit definition
│   ├── circuits.py              # Additional circuit architectures
│   └── quantum_layer.py         # Quantum feature extraction utilities
│
├── models/
│   ├── classical.py             # Classical model training and evaluation
│   ├── classical_models.py      # Classical model definitions
│   ├── hybrid.py                # Hybrid model construction
│   └── hybrid_model.py          # Hybrid model definitions
│
├── preprocessing/
│   ├── feature_extraction.py    # Feature engineering
│   ├── features.py              # Feature definitions
│   ├── features_nordkit.py      # RDKit-free features
│   └── pipeline.py              # Full preprocessing pipeline
│
├── evaluation/
│   ├── metrics.py               # Evaluation metrics
│   ├── visualize.py             # Visualization utilities
│   └── visualizer.py            # Extended visualizer
│
├── data/
│   ├── loader.py                # Data loading utilities
│   └── qm9.csv                  # Raw QM9 subset
│
├── dashboard/
│   ├── app.py                   # Dashboard backend
│   └── index.html               # Dashboard frontend
│
├── results/                     # Generated plots and result CSVs
├── results_advanced/            # Advanced experiment results
├── notebooks/                   # Experiment notebooks
└── requirements.txt             # Python dependencies
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/SHARMI-P/Quantum-Classical-Hybrid-Machine-Learning-for-Molecular-Property-Prediction.git
cd Quantum-Classical-Hybrid-Machine-Learning-for-Molecular-Property-Prediction

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Full Pipeline

```bash
python build_model.py
```

This executes **7 sequential pipeline steps**:

| Step | Description |
|---|---|
| 1 | Load QM9 data — parse CSV, convert units, report statistics |
| 2 | 3-way split — Train (70%) / Validation (15%) / Test (15%) |
| 3 | VQC training — Adam optimizer, 300 steps, batch=80 |
| 4 | Feature extraction — quantum features for all splits |
| 5 | GridSearchCV tuning — all 7 models with 5-fold CV |
| 6 | Repeated 5×2 CV — statistical comparison (classical vs hybrid) |
| 7 | Plot generation — 5 publication-quality plots saved to `results/` |

### Run Individual Experiments

```bash
python run_experiment.py       # Standard experiment
python run_simulation.py       # Quantum circuit simulation
python advanced_experiment.py  # Extended ablation study
```

---

## 🔬 Methodology

### Variational Quantum Circuit (VQC)

- **Qubits:** 6 (one per encoded feature)
- **Layers:** 2 variational layers
- **Trainable Parameters:** 24 (2 layers × 6 qubits × 2 rotation angles)
- **Encoding:** Angle encoding — RY(xᵢ × π) gates
- **Entanglement:** CNOT ladder + wrap-around CNOT
- **Measurement:** PauliZ expectation values → 6D quantum feature vector
- **Optimizer:** Adam (lr = 0.03), 300 steps, batch size = 80

### Hybrid Feature Construction

```
Classical Features (18D)  +  Quantum Features (6D)  →  Hybrid Features (24D)
```

### Models Evaluated

| Model | Type |
|---|---|
| Ridge Regression | Classical |
| ElasticNet | Classical |
| SVR (RBF kernel) | Classical |
| Random Forest | Classical |
| Hybrid Ridge | Quantum-Classical |
| Hybrid SVR | Quantum-Classical |
| Hybrid Random Forest | Quantum-Classical |

---

## 📊 Results

### Test Set Performance

| Model | Type | RMSE (eV) | R² | Rank |
|---|---|---|---|---|
| **Random Forest** | Classical | **0.3328** | **0.7676** | 1st ★ |
| SVR (tuned) | Classical | 0.3337 | 0.7664 | 2nd |
| Hybrid SVR | Hybrid Q-C | 0.3448 | 0.7506 | 3rd |
| Hybrid RF | Hybrid Q-C | 0.3554 | 0.7350 | 4th |
| Hybrid Ridge | Hybrid Q-C | 0.4472 | 0.5804 | 5th |
| Ridge | Classical | 0.4489 | 0.5773 | 6th |
| ElasticNet | Classical | 0.4650 | 0.5463 | 7th |

### Statistical Comparison (Classical SVR vs Hybrid SVR)

| Metric | Value |
|---|---|
| Classical SVR Mean RMSE | 0.3626 ± 0.0520 eV |
| Hybrid SVR Mean RMSE | 0.3653 ± 0.0499 eV |
| Mean Improvement | −2.65 meV (Hybrid slightly worse) |
| Paired t-test | t = −1.61, **p = 0.1411** (not significant) |
| Wilcoxon Signed-Rank | W = 12.0, **p = 0.1309** (not significant) |
| Cohen's d | −0.538 (moderate effect against hybrid) |

> **Key Finding:** Classical models outperformed hybrid models. The difference is not statistically significant (p > 0.05), consistent with current literature on NISQ-era quantum ML limitations.

---

## 📈 Generated Plots

All plots are saved to the `results/` directory:

| File | Description |
|---|---|
| `01_vqc_training_loss.png` | VQC Adam convergence curve with best validation loss marker |
| `02_cv_comparison.png` | Classical vs Hybrid SVR RMSE bar chart with error bars |
| `03_homo_distribution.png` | Histogram of QM9 HOMO energies (eV) |
| `04_all_models.png` | RMSE and R² comparison across all 7 models |
| `05_fold_comparison.png` | Per-fold RMSE for Classical vs Hybrid SVR (10 folds) |

---

## 🧠 Discussion

The hybrid models did not outperform classical baselines. This is explained by:

- **Feature redundancy** — the 18 classical descriptors already capture chemically relevant HOMO information; the VQC only re-encodes a subset (6 features)
- **Circuit expressibility** — 6 qubits, 2 layers, 24 parameters is limited for this task
- **NISQ limitations** — current VQCs lack the exponential advantage of fault-tolerant quantum computers
- **Dataset scale** — 2,000 molecules may be insufficient to expose quantum-learnable structure

This result is consistent with Cerezo et al. (2021), Schuld & Petruccione (2021), and Huang et al. (2021).

### What Would Improve Hybrid Performance

- More qubits (9–12) to encode all 18 features directly
- Deeper circuits for improved expressibility
- Data re-uploading strategy (Perez-Salinas et al., 2020)
- Quantum-native molecular encodings (graph/SMILES-based)
- Training on the full QM9 dataset (134k molecules)
- Multi-target prediction (HOMO + LUMO + dipole moment)

---

## 🔧 Configuration

Key hyperparameters are defined in `config.py`:

| Parameter | Value |
|---|---|
| VQC qubits | 6 |
| VQC layers | 2 |
| VQC learning rate | 0.03 |
| VQC training steps | 300 |
| Batch size | 80 |
| Random seed | 42 |
| Train / Val / Test split | 70% / 15% / 15% |

### Optimal Hyperparameters (Found via GridSearchCV)

| Model | Best Parameters |
|---|---|
| Ridge | alpha = 0.01 |
| ElasticNet | alpha = 0.01, l1_ratio = 0.3 |
| SVR | C = 10.0, epsilon = 0.1, gamma = 'auto' |
| Random Forest | max_depth = None, n_estimators = 200 |
| Hybrid Ridge | alpha = 0.01 |
| Hybrid SVR | C = 10.0, epsilon = 0.1, gamma = 'auto' |
| Hybrid RF | max_depth = None, n_estimators = 200 |

---

## 🔭 Future Work

- [ ] Extend to full QM9 dataset (134k molecules) with GPU-accelerated simulation
- [ ] Implement data re-uploading circuits (Perez-Salinas et al., 2020)
- [ ] Multi-property prediction (HOMO, LUMO, dipole moment, polarizability)
- [ ] Graph-based quantum encodings using molecular topology
- [ ] Benchmark on real quantum hardware (IBM Quantum, IonQ)

---

## 📚 References

1. Ramakrishnan et al. (2014). QM9 dataset. *Scientific Data*, 1, 140022.
2. Cerezo, M., et al. (2021). Variational quantum algorithms. *Nature Reviews Physics*, 3(9), 625–644.
3. Huang, H. Y., et al. (2021). Power of data in quantum machine learning. *Nature Communications*, 12(1), 2631.
4. Perez-Salinas, A., et al. (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.
5. Bergholm, V., et al. (2022). PennyLane: Automatic differentiation of hybrid quantum-classical computations. arXiv:1811.04968.
6. Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum Computers*. Springer.
7. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923.

---

*Report Date: April 15, 2026 | Framework: PennyLane + scikit-learn | Dataset: QM9*