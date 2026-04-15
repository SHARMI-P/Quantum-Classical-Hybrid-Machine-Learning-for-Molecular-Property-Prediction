"""
config.py  –  Central configuration for the project.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
N_SAMPLES    = 300          # manageable subset
RANDOM_SEED  = 42
TEST_SIZE    = 0.20

# ── Target property to predict ────────────────────────────────────────────────
# Options: "homo", "lumo", "gap", "zpve", "alpha", "mu"
TARGET       = "homo"

# ── Feature engineering ───────────────────────────────────────────────────────
MORGAN_BITS  = 128          # fingerprint length
MORGAN_RADIUS= 2
USE_PCA      = True
N_PCA        = 12

# ── Quantum circuit ───────────────────────────────────────────────────────────
N_QUBITS     = 4
N_Q_LAYERS   = 2

# ── Hybrid model ──────────────────────────────────────────────────────────────
N_CLASSICAL_EXTRA = 8
