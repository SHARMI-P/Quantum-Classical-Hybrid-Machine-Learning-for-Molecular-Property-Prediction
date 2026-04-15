# preprocessing/pipeline.py — Full Preprocessing Pipeline
"""
End-to-end preprocessing pipeline:
  SMILES → Features → Normalize → PCA → Train/Val/Test splits

The PCA compression step is CRITICAL for quantum encoding:
  - Quantum angle encoding requires exactly N_QUBITS features
  - PCA reduces from ~300+ dims to 8 dims (one per qubit)
  - StandardScaler ensures features are in [−π, π] range after scaling
"""

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_PCA_COMPONENTS, TEST_SIZE, VAL_SIZE, RANDOM_STATE,
                    CACHE_DIR, USE_RDKIT_DESCRIPTORS)
from preprocessing.feature_extraction import extract_features_batch

logger = logging.getLogger(__name__)


class MolecularPreprocessor:
    """
    Full preprocessing pipeline for molecular property prediction.
    
    Steps:
    1. Feature extraction (Morgan FP + RDKit descriptors)
    2. StandardScaler normalization
    3. PCA compression to N_PCA_COMPONENTS
    4. Train/validation/test splitting
    
    The fitted scaler and PCA are preserved for inference on new molecules.
    """

    def __init__(self, n_components: int = N_PCA_COMPONENTS):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        self.is_fitted = False
        self.feature_dim_raw = None
        self.explained_variance_ratio = None

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "gap",
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Fit the pipeline on training data and return split datasets.
        
        Args:
            df: DataFrame with 'smiles' column and target property column
            target_col: Name of the property to predict
            
        Returns:
            splits: dict with keys 'X_train', 'X_val', 'X_test' (PCA-reduced)
            splits_raw: dict with same keys but raw (pre-PCA) features
        """
        logger.info("=" * 60)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 60)

        # ─── Step 1: Extract raw features ────────────────────────────────────
        logger.info("Step 1/4: Extracting molecular features...")
        smiles_list = df["smiles"].tolist()
        X_raw, valid_idx = extract_features_batch(smiles_list, USE_RDKIT_DESCRIPTORS)
        y = df[target_col].values[valid_idx]

        self.feature_dim_raw = X_raw.shape[1]
        logger.info(f"  Raw feature shape: {X_raw.shape}")
        logger.info(f"  Target shape: {y.shape}")

        # ─── Step 2: Split BEFORE fitting scaler (prevent data leakage) ───────
        logger.info("Step 2/4: Splitting into train/val/test...")
        val_relative = VAL_SIZE / (1 - TEST_SIZE)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative, random_state=RANDOM_STATE
        )

        logger.info(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        # ─── Step 3: Normalize ─────────────────────────────────────────────────
        logger.info("Step 3/4: Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # ─── Step 4: PCA compression ───────────────────────────────────────────
        logger.info(f"Step 4/4: PCA compression ({self.feature_dim_raw} → {self.n_components} dims)...")
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)

        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        total_var = self.explained_variance_ratio.sum()
        logger.info(f"  Explained variance ({self.n_components} components): {total_var:.1%}")

        # ─── Scale PCA output to [−π, π] for quantum encoding ─────────────────
        self.pca_scaler = StandardScaler()
        X_train_pca = self.pca_scaler.fit_transform(X_train_pca)
        X_val_pca = self.pca_scaler.transform(X_val_pca)
        X_test_pca = self.pca_scaler.transform(X_test_pca)

        # Clip to quantum-friendly range
        clip_val = np.pi
        X_train_pca = np.clip(X_train_pca, -clip_val, clip_val)
        X_val_pca = np.clip(X_val_pca, -clip_val, clip_val)
        X_test_pca = np.clip(X_test_pca, -clip_val, clip_val)

        self.is_fitted = True

        splits_pca = {
            "X_train": X_train_pca.astype(np.float32),
            "X_val": X_val_pca.astype(np.float32),
            "X_test": X_test_pca.astype(np.float32),
            "y_train": y_train.astype(np.float32),
            "y_val": y_val.astype(np.float32),
            "y_test": y_test.astype(np.float32),
        }

        splits_raw = {
            "X_train": X_train_scaled.astype(np.float32),
            "X_val": X_val_scaled.astype(np.float32),
            "X_test": X_test_scaled.astype(np.float32),
            "y_train": y_train.astype(np.float32),
            "y_val": y_val.astype(np.float32),
            "y_test": y_test.astype(np.float32),
        }

        logger.info("Preprocessing complete!")
        return splits_pca, splits_raw

    def transform_new(self, smiles_list) -> np.ndarray:
        """Transform new SMILES at inference time."""
        assert self.is_fitted, "Pipeline not fitted yet"
        X_raw, _ = extract_features_batch(smiles_list, USE_RDKIT_DESCRIPTORS)
        X_scaled = self.scaler.transform(X_raw)
        X_pca = self.pca.transform(X_scaled)
        X_pca = self.pca_scaler.transform(X_pca)
        return np.clip(X_pca, -np.pi, np.pi).astype(np.float32)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self):
        """Print preprocessing summary."""
        print("\n" + "=" * 50)
        print("PREPROCESSING PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Raw feature dimensions: {self.feature_dim_raw}")
        print(f"PCA components:         {self.n_components}")
        if self.explained_variance_ratio is not None:
            print(f"Explained variance:     {self.explained_variance_ratio.sum():.1%}")
            print(f"\nPer-component variance:")
            for i, v in enumerate(self.explained_variance_ratio):
                bar = "█" * int(v * 50)
                print(f"  PC{i+1:2d}: {v:.3f} {bar}")
        print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.loader import load_dataset

    df = load_dataset(n_molecules=200)
    preprocessor = MolecularPreprocessor(n_components=8)
    splits_pca, splits_raw = preprocessor.fit_transform(df, target_col="gap")

    preprocessor.summary()
    print(f"\nPCA splits:")
    for k, v in splits_pca.items():
        print(f"  {k}: {v.shape}")
