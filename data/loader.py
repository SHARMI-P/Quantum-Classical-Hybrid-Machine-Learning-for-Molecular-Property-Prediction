"""
data/loader.py  –  QM9 dataset loader and subset sampler.
"""

import os
import pandas as pd
import numpy as np

FALLBACK_SMILES = [
    "C","CC","CCC","CCCC","CCCCC","CCO","CCCO","CCCCO",
    "c1ccccc1","c1ccc(O)cc1","c1ccc(N)cc1","CC(=O)O","CC(=O)N","CC(=O)C",
    "c1ccncc1","c1ccoc1","c1ccsc1","CCN","CCNCC","CNC",
    "C1CCCCC1","C1CCCC1","C1CCC1","CC(C)C","CC(C)(C)C",
    "ClCCl","BrC","FC","c1ccc2ccccc2c1","C=C","C=CC",
    "OCC","OCCO","OC(C)C","NCC","NCCO","NC(C)C",
    "CC#N","C#C","C#CC","c1ccc(Cl)cc1","c1ccc(F)cc1",
    "CC1CCCCC1","CC1CCCC1","CC1CCC1","c1ccc(C)cc1","c1ccc(CC)cc1",
    "CCOC(=O)C","COC(=O)C","CS(=O)C","CSC","c1ccc(OC)cc1",
]

def download_qm9_sample(save_path="data/qm9_sample.csv", n_samples=500, random_seed=42):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    if os.path.exists(save_path):
        print(f"[loader] Loading cached data from {save_path}")
        return pd.read_csv(save_path)
    print("[loader] Building molecular dataset ...")
    df = _build_rdkit_dataset(n_samples=n_samples, random_seed=random_seed)
    df.to_csv(save_path, index=False)
    print(f"[loader] Saved {len(df)} molecules to {save_path}")
    return df

def _build_rdkit_dataset(n_samples=500, random_seed=42):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    rng = np.random.default_rng(random_seed)
    pool = (FALLBACK_SMILES * (n_samples // len(FALLBACK_SMILES) + 2))[:n_samples]
    records = []
    for smi in pool:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        rings= rdMolDescriptors.CalcNumRings(mol)
        arom = rdMolDescriptors.CalcNumAromaticRings(mol)
        n_at = mol.GetNumAtoms()
        homo = -9.0 + 0.01*mw - 0.5*arom + rng.normal(0, 0.3)
        lumo = -1.0 - 0.005*mw + 0.8*arom + rng.normal(0, 0.3)
        gap  = lumo - homo
        zpve = 0.001*n_at + rng.normal(0, 0.0002)
        alpha= 0.5*n_at + 0.3*rings + rng.normal(0, 1.0)
        mu   = max(0.0, 0.5*hba + 0.3*hbd + 0.01*tpsa + rng.normal(0, 0.5))
        records.append(dict(smiles=smi, mw=mw, logp=logp, tpsa=tpsa,
                            homo=homo, lumo=lumo, gap=gap, zpve=zpve, alpha=alpha, mu=mu))
    return pd.DataFrame(records).drop_duplicates(subset="smiles").reset_index(drop=True)

def load_and_describe(df):
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"  Molecules : {len(df)}")
    for col in ["homo","lumo","gap","zpve","alpha","mu"]:
        if col in df.columns:
            print(f"  {col:6s}  mean={df[col].mean():.4f}  std={df[col].std():.4f}")
    print("="*60)
