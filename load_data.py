import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import os

os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ── Load QM9 ──
df = pd.read_csv('data/qm9.csv')
print(f"Loaded {len(df)} molecules")

# ── Feature extraction function ──
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            # Basic descriptors (improved versions of what you had)
            'mol_weight':       Descriptors.MolWt(mol),
            'logP':             Descriptors.MolLogP(mol),
            'tpsa':             Descriptors.TPSA(mol),
            'num_rotatable':    rdMolDescriptors.CalcNumRotatableBonds(mol),
            'num_hdonors':      rdMolDescriptors.CalcNumHBD(mol),
            'num_hacceptors':   rdMolDescriptors.CalcNumHBA(mol),
            'num_rings':        rdMolDescriptors.CalcNumRings(mol),
            'num_aromatic':     rdMolDescriptors.CalcNumAromaticRings(mol),
            'num_heavy_atoms':  mol.GetNumHeavyAtoms(),
            'num_heteroatoms':  rdMolDescriptors.CalcNumHeteroatoms(mol),
            # Quantum-chemistry relevant descriptors (NEW)
            'hall_kier_alpha':  Descriptors.HallKierAlpha(mol),
            'kappa1':           Descriptors.Kappa1(mol),
            'kappa2':           Descriptors.Kappa2(mol),
            'chi0v':            Descriptors.Chi0v(mol),
            'chi1v':            Descriptors.Chi1v(mol),
            'bertz_ct':         Descriptors.BertzCT(mol),
            'num_valence_e':    sum(
                                    atom.GetTotalValence()
                                    for atom in mol.GetAtoms()
                                ),
            'max_partial_charge': max(
                                    float(atom.GetNoImplicit())
                                    for atom in mol.GetAtoms()
                                ),
        }
    except Exception:
        return None

# ── Process 2000 molecules ──
# (enough for academic proof-of-concept, manageable runtime)
N = 2000
records = []
skipped = 0

print(f"Extracting features from {N} molecules...")
for i in range(N):
    row = df.iloc[i]
    feats = extract_features(row['smiles'])
    if feats is not None:
        feats['homo_energy'] = row['homo']   # Real DFT target (eV)
        feats['smiles']      = row['smiles']
        records.append(feats)
    else:
        skipped += 1
    if (i+1) % 500 == 0:
        print(f"  Processed {i+1}/{N}...")

# ── Save ──
out = pd.DataFrame(records)
out.to_csv('data/qm9_features.csv', index=False)

print(f"\n--- Feature Extraction Complete ---")
print(f"Molecules processed : {len(out)}  (skipped {skipped} unparseable)")
print(f"Features per molecule: {len(out.columns)-2}  (excluding homo + smiles)")
print(f"\nHOMO energy stats (real DFT values):")
print(f"  Min  : {out['homo_energy'].min():.4f} eV")
print(f"  Max  : {out['homo_energy'].max():.4f} eV")
print(f"  Mean : {out['homo_energy'].mean():.4f} eV")
print(f"  Std  : {out['homo_energy'].std():.4f} eV")
print(f"\nSaved to data/qm9_features.csv")
print("Phase 1 Step 2 COMPLETE!")