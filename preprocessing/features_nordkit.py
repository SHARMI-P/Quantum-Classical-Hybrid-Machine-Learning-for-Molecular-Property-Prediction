"""
preprocessing/features_nordkit.py
===================================
Fallback featurizer that works WITHOUT RDKit.
Uses SMILES string statistics as proxy molecular descriptors.
Used only for running demonstrations in restricted environments.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def smiles_descriptors(smi: str) -> dict:
    """Compute hand-crafted descriptors from SMILES string characters."""
    return {
        "length":       len(smi),
        "n_carbons":    smi.count("C") + smi.count("c"),
        "n_nitrogens":  smi.count("N") + smi.count("n"),
        "n_oxygens":    smi.count("O") + smi.count("o"),
        "n_sulfurs":    smi.count("S") + smi.count("s"),
        "n_halogens":   smi.count("F") + smi.count("Cl") + smi.count("Br"),
        "n_aromatic":   smi.count("c") + smi.count("n") + smi.count("o") + smi.count("s"),
        "n_rings":      smi.count("1") + smi.count("2") + smi.count("3"),
        "n_double":     smi.count("="),
        "n_triple":     smi.count("#"),
        "n_branches":   smi.count("("),
        "n_pos_charge": smi.count("+"),
        "n_neg_charge": smi.count("-"),
        "has_aromatic": float("c" in smi or "n" in smi),
        "frac_aromatic":sum(1 for c in smi if c.islower()) / max(len(smi), 1),
        "mw_approx":    (smi.count("C")+smi.count("c"))*12 +
                        (smi.count("N")+smi.count("n"))*14 +
                        (smi.count("O")+smi.count("o"))*16 +
                        (smi.count("S")+smi.count("s"))*32,
    }


def smiles_fingerprint(smi: str, n_bits: int = 64) -> np.ndarray:
    """Hashed SMILES n-gram fingerprint as a binary bit vector."""
    fp = np.zeros(n_bits, dtype=np.float32)
    for n in [2, 3, 4]:
        for i in range(len(smi) - n + 1):
            gram = smi[i:i+n]
            idx  = hash(gram) % n_bits
            fp[idx] = 1.0
    return fp


class MolecularFeaturizer:
    """Drop-in replacement featurizer when RDKit is absent."""

    def __init__(self, n_bits=64, radius=2, use_pca=True, n_pca_components=8):
        self.n_bits = n_bits
        self.radius = radius
        self.use_pca = use_pca
        self.n_pca_components = n_pca_components
        self.scaler = StandardScaler()
        self.pca    = None
        self._fitted = False

    def _featurize(self, smiles_list):
        descs = []
        fps   = []
        valid_idx = []
        for i, smi in enumerate(smiles_list):
            d   = smiles_descriptors(smi)
            fp  = smiles_fingerprint(smi, self.n_bits)
            descs.append(list(d.values()))
            fps.append(fp)
            valid_idx.append(i)
        return np.array(descs, dtype=np.float32), np.array(fps), valid_idx

    def fit_transform(self, smiles_list):
        desc, fps, valid_idx = self._featurize(smiles_list)
        desc_scaled = self.scaler.fit_transform(desc)
        n = min(self.n_pca_components, desc_scaled.shape[0]-1, desc_scaled.shape[1])
        if self.use_pca and n > 1:
            self.pca = PCA(n_components=n)
            desc_r = self.pca.fit_transform(desc_scaled)
        else:
            desc_r = desc_scaled
        X = np.hstack([desc_r, fps])
        self._fitted = True
        print(f"[featurizer] {len(valid_idx)} molecules → feature shape {X.shape}")
        return X, valid_idx

    def transform(self, smiles_list):
        desc, fps, valid_idx = self._featurize(smiles_list)
        desc_scaled = self.scaler.transform(desc)
        desc_r = self.pca.transform(desc_scaled) if self.pca else desc_scaled
        return np.hstack([desc_r, fps]), valid_idx

    def quantum_features(self, X, n_qubits=4):
        variances = np.var(X, axis=0)
        top_idx   = np.argsort(variances)[-n_qubits:][::-1]
        X_top = X[:, top_idx]
        mins  = X_top.min(axis=0, keepdims=True)
        maxs  = X_top.max(axis=0, keepdims=True)
        rng   = np.where(maxs - mins < 1e-8, 1.0, maxs - mins)
        X_q   = (X_top - mins) / rng * np.pi
        return X_q, top_idx
