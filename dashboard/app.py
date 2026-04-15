"""
dashboard/app.py
================
Streamlit dashboard for interactive inference.
Run with:  streamlit run dashboard/app.py
"""
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

def main():
    if not HAS_STREAMLIT:
        print("Install streamlit: pip install streamlit")
        return

    st.set_page_config(page_title="Quantum Mol Predictor", layout="wide")
    st.title("⚛️  Quantum-Enhanced Molecular Property Predictor")
    st.markdown("Predict HOMO/LUMO energy using a Hybrid Quantum-Classical ML model.")

    col1, col2, col3 = st.columns(3)
    with col1:
        mw   = st.slider("Molecular Weight (g/mol)", 16.0, 300.0, 78.0)
        logp = st.slider("LogP", -3.0, 6.0, 1.5)
    with col2:
        tpsa = st.slider("TPSA (Å²)", 0.0, 140.0, 20.0)
        hba  = st.slider("H-Bond Acceptors", 0, 8, 1)
    with col3:
        hbd  = st.slider("H-Bond Donors", 0, 5, 0)
        arom = st.slider("Aromatic Rings", 0, 3, 1)

    features = np.array([[mw, logp, tpsa, hba, hbd, arom]])

    # Simple linear approximation for the dashboard
    homo_approx = -9.0 + 0.008*mw - 0.4*arom + 0.1*logp
    lumo_approx = -1.0 - 0.004*mw + 0.7*arom - 0.1*logp
    gap_approx  = lumo_approx - homo_approx

    st.divider()
    st.subheader("Predicted Properties")
    c1, c2, c3 = st.columns(3)
    c1.metric("HOMO Energy", f"{homo_approx:.3f} eV")
    c2.metric("LUMO Energy", f"{lumo_approx:.3f} eV")
    c3.metric("HOMO-LUMO Gap", f"{gap_approx:.3f} eV")

    st.info("For full accuracy, run the complete pipeline via run_experiment.py")

if __name__ == "__main__":
    main()
