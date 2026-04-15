"""
preprocessing/features.py
==========================
Convert SMILES strings → numerical feature vectors using RDKit.

Two representations are produced:
  1. Descriptor vector  – 200-D physicochemical descriptors (continuous)
  2. Morgan fingerprint – 1024-bit circular fingerprint (binary)

Both are combined into a single feature matrix used by downstream models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MolecularFeaturizer:
    """
    Featurize a list of SMILES into descriptor + fingerprint arrays.

    Parameters
    ----------
    n_bits : int
        Length of Morgan fingerprint bit vector.
    radius : int
        Morgan algorithm radius (2 = ECFP4).
    use_pca : bool
        Whether to reduce descriptor dimensionality with PCA.
    n_pca_components : int
        Number of PCA components to keep (only if use_pca=True).
    """

    def __init__(self, n_bits=256, radius=2, use_pca=True, n_pca_components=16):
        self.n_bits = n_bits
        self.radius = radius
        self.use_pca = use_pca
        self.n_pca_components = n_pca_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components) if use_pca else None
        self._fitted = False

    # ------------------------------------------------------------------
    def _smiles_to_mol(self, smiles_list):
        from rdkit import Chem
        mols, valid_idx = [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                valid_idx.append(i)
        return mols, valid_idx

    def _compute_descriptors(self, mols):
        """Return a (n_mols, n_desc) array of RDKit physicochemical descriptors."""
        from rdkit.Chem import Descriptors
        desc_names = [
            "MolWt", "MolLogP", "MolMR", "TPSA", "NumHAcceptors",
            "NumHDonors", "NumRotatableBonds", "NumAromaticRings",
            "NumAliphaticRings", "NumSaturatedRings", "FractionCSP3",
            "HeavyAtomCount", "NHOHCount", "NOCount", "NumHeteroatoms",
            "RingCount", "BalabanJ", "BertzCT", "Chi0", "Chi0n",
            "Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v",
            "Kappa1", "Kappa2", "Kappa3",
        ]
        rows = []
        for mol in mols:
            row = []
            for name in desc_names:
                try:
                    val = getattr(Descriptors, name)(mol)
                    row.append(float(val) if val is not None and np.isfinite(float(val)) else 0.0)
                except Exception:
                    row.append(0.0)
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    def _compute_fingerprints(self, mols):
        """Return a (n_mols, n_bits) binary array of Morgan fingerprints."""
        from rdkit.Chem import AllChem
        fps = []
        for mol in mols:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            fps.append(np.array(fp, dtype=np.float32))
        return np.array(fps)

    # ------------------------------------------------------------------
    def fit_transform(self, smiles_list):
        """
        Fit the scaler (and optional PCA) on the descriptors, then
        return the combined feature matrix and the valid-index mask.
        """
        mols, valid_idx = self._smiles_to_mol(smiles_list)
        desc  = self._compute_descriptors(mols)
        fps   = self._compute_fingerprints(mols)

        # Scale descriptors
        desc_scaled = self.scaler.fit_transform(desc)
        if self.use_pca:
            n = min(self.n_pca_components, desc_scaled.shape[0], desc_scaled.shape[1])
            self.pca = PCA(n_components=n)
            desc_reduced = self.pca.fit_transform(desc_scaled)
        else:
            desc_reduced = desc_scaled

        X = np.hstack([desc_reduced, fps])
        self._fitted = True
        print(f"[featurizer] {len(mols)} valid molecules → feature shape {X.shape}")
        return X, valid_idx

    def transform(self, smiles_list):
        """Transform new SMILES using the fitted scaler/PCA."""
        assert self._fitted, "Call fit_transform first."
        mols, valid_idx = self._smiles_to_mol(smiles_list)
        desc  = self._compute_descriptors(mols)
        fps   = self._compute_fingerprints(mols)
        desc_scaled = self.scaler.transform(desc)
        desc_reduced = self.pca.transform(desc_scaled) if self.use_pca else desc_scaled
        return np.hstack([desc_reduced, fps]), valid_idx

    def quantum_features(self, X, n_qubits=4):
        """
        Select and normalize the top-n_qubits most-variant features
        into [0, π] for angle encoding into a quantum circuit.
        """
        variances = np.var(X, axis=0)
        top_idx   = np.argsort(variances)[-n_qubits:][::-1]
        X_top = X[:, top_idx]
        # min-max scale to [0, pi]
        mins  = X_top.min(axis=0, keepdims=True)
        maxs  = X_top.max(axis=0, keepdims=True)
        rng   = np.where(maxs - mins < 1e-8, 1.0, maxs - mins)
        X_q   = (X_top - mins) / rng * np.pi
        return X_q, top_idx
