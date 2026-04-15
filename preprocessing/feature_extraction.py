# preprocessing/feature_extraction.py — RDKit Molecular Feature Extraction
"""
Converts SMILES strings into numerical feature vectors using RDKit.

Two complementary representations:
1. Morgan Fingerprints (circular fingerprints): Encode local atomic environments
   up to a given radius. Captures structural patterns (rings, functional groups).

2. RDKit Molecular Descriptors: 196+ physicochemical properties including
   molecular weight, logP, TPSA, number of H-bond donors/acceptors, etc.

Final feature vector = [fingerprints | descriptors], standardized to zero mean / unit variance.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MORGAN_RADIUS, MORGAN_NBITS, USE_RDKIT_DESCRIPTORS

logger = logging.getLogger(__name__)

# ─── RDKit imports ────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")  # Suppress RDKit warnings
    RDKIT_AVAILABLE = True
    logger.info("RDKit loaded successfully")
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available — using fallback feature extraction")


# ─── Descriptor names ─────────────────────────────────────────────────────────
SELECTED_DESCRIPTORS = [
    "MolWt", "ExactMolWt", "HeavyAtomMolWt",
    "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
    "NumRotatableBonds", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticRings", "NumAromaticHeterocycles",
    "NumSaturatedHeterocycles", "NumAliphaticHeterocycles",
    "NumSaturatedCarbocycles", "NumAromaticCarbocycles",
    "NumAliphaticCarbocycles", "RingCount",
    "MolLogP", "MolMR", "TPSA",
    "LabuteASA", "BalabanJ", "BertzCT",
    "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v",
    "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",
    "HallKierAlpha", "Ipc", "Kappa1", "Kappa2", "Kappa3",
    "MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex",
    "MinEStateIndex", "FractionCSP3", "HeavyAtomCount",
    "NHOHCount", "NOCount",
    "NumRadicalElectrons", "NumValenceElectrons",
    "qed",  # Drug-likeness score
]


def smiles_to_mol(smiles: str):
    """Parse SMILES to RDKit mol object with sanitization."""
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def compute_morgan_fingerprint(mol, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_NBITS) -> np.ndarray:
    """
    Compute Morgan (ECFP) fingerprint as a bit vector.
    
    Morgan fingerprints encode the local chemical environment of each atom
    out to a given radius. radius=2 ≈ ECFP4, radius=3 ≈ ECFP6.
    """
    if mol is None:
        return np.zeros(n_bits)
    try:
        fp = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return np.zeros(n_bits)


def compute_descriptors(mol) -> np.ndarray:
    """
    Compute a curated set of RDKit molecular descriptors.
    
    Returns a fixed-length vector of physicochemical properties.
    NaN values are replaced with 0.
    """
    if mol is None or not RDKIT_AVAILABLE:
        return np.zeros(len(SELECTED_DESCRIPTORS))

    values = []
    for desc_name in SELECTED_DESCRIPTORS:
        try:
            fn = getattr(Descriptors, desc_name, None)
            if fn is None:
                # Try rdMolDescriptors
                fn = getattr(rdMolDescriptors, desc_name, None)
            if fn is not None:
                val = fn(mol)
                values.append(float(val) if val is not None else 0.0)
            else:
                values.append(0.0)
        except Exception:
            values.append(0.0)

    arr = np.array(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def smiles_to_features(smiles: str, use_descriptors: bool = USE_RDKIT_DESCRIPTORS) -> Optional[np.ndarray]:
    """
    Convert a single SMILES string to a feature vector.
    
    Returns:
        np.ndarray of shape (n_bits + n_descriptors,) or None if invalid
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    fp = compute_morgan_fingerprint(mol)

    if use_descriptors:
        desc = compute_descriptors(mol)
        features = np.concatenate([fp, desc])
    else:
        features = fp

    return features.astype(np.float32)


def extract_features_batch(
    smiles_list: List[str],
    use_descriptors: bool = USE_RDKIT_DESCRIPTORS,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract features for a batch of SMILES strings.
    
    Returns:
        X: np.ndarray of shape (n_valid, n_features)
        valid_indices: list of indices where extraction succeeded
    """
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available — using random features as fallback")
        n = len(smiles_list)
        dim = MORGAN_NBITS + (len(SELECTED_DESCRIPTORS) if use_descriptors else 0)
        return np.random.randn(n, dim).astype(np.float32), list(range(n))

    features, valid_indices = [], []
    n_failed = 0

    for i, smi in enumerate(smiles_list):
        if verbose and i % 500 == 0:
            logger.info(f"  Processing molecule {i}/{len(smiles_list)}...")

        feat = smiles_to_features(smi, use_descriptors)
        if feat is not None:
            features.append(feat)
            valid_indices.append(i)
        else:
            n_failed += 1

    if n_failed > 0:
        logger.warning(f"Feature extraction failed for {n_failed} molecules")

    X = np.array(features, dtype=np.float32)
    logger.info(f"Extracted features: shape={X.shape}, valid={len(valid_indices)}")
    return X, valid_indices


def get_feature_names(use_descriptors: bool = USE_RDKIT_DESCRIPTORS) -> List[str]:
    """Return feature names for interpretability."""
    names = [f"morgan_{i}" for i in range(MORGAN_NBITS)]
    if use_descriptors:
        names += SELECTED_DESCRIPTORS
    return names


# ─── Fallback feature extraction (no RDKit) ───────────────────────────────────
def smiles_to_features_fallback(smiles: str, dim: int = 64) -> np.ndarray:
    """
    Character-level hash-based feature extraction.
    Used when RDKit is unavailable. Not chemically meaningful.
    """
    np.random.seed(hash(smiles) % (2**31))
    feat = np.random.randn(dim).astype(np.float32)
    # Encode some structural info via character counting
    feat[0] = smiles.count('c') / max(len(smiles), 1)   # aromaticity
    feat[1] = smiles.count('N') / max(len(smiles), 1)   # nitrogen
    feat[2] = smiles.count('O') / max(len(smiles), 1)   # oxygen
    feat[3] = smiles.count('=') / max(len(smiles), 1)   # double bonds
    feat[4] = smiles.count('#') / max(len(smiles), 1)   # triple bonds
    feat[5] = smiles.count('1') + smiles.count('2')     # rings
    feat[6] = len(smiles) / 100.0                        # molecule size
    return feat


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_smiles = ["c1ccccc1", "CC=O", "C1CCCCC1", "c1ccncc1", "CCO"]
    print(f"\nRDKit available: {RDKIT_AVAILABLE}")
    print(f"Feature dimensions:")
    print(f"  Morgan fingerprints: {MORGAN_NBITS}")
    print(f"  Descriptors: {len(SELECTED_DESCRIPTORS)}")
    print(f"  Total: {MORGAN_NBITS + len(SELECTED_DESCRIPTORS)}")

    X, valid = extract_features_batch(test_smiles)
    print(f"\nExtracted X shape: {X.shape}")
    print(f"Valid indices: {valid}")
    print(f"\nBenzene fingerprint (first 20 bits): {X[0, :20]}")
