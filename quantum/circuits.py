# quantum/circuits.py — Variational Quantum Circuits using PennyLane
"""
Implements parameterized quantum circuits (PQC) for molecular feature encoding.

Two encoding strategies:
1. Angle Encoding: each feature → rotation angle on a qubit (RY gate)
   - Simple, preserves sign information
   - Requires n_features = n_qubits (use PCA first)

2. Amplitude Encoding: feature vector → quantum state amplitudes
   - More information-dense (2^n_qubits amplitudes from n_qubits)
   - Requires normalization (||x|| = 1)

Variational Ansatz:
   - Strongly Entangling Layers (Rot + CNOT ladder)
   - Parameters: 3 × n_qubits × n_layers (Rot gate has 3 parameters)

Measurement:
   - PauliZ expectation values on each qubit
   - Output: n_qubits real numbers in [−1, +1]
   - These become features for the classical part
"""

import os
import sys
import numpy as np
import logging
from typing import Callable, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS, N_LAYERS, QUANTUM_DEVICE

logger = logging.getLogger(__name__)

# ─── PennyLane import ─────────────────────────────────────────────────────────
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
    logger.info(f"PennyLane {qml.version()} loaded successfully")
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not available — quantum circuits will use simulation fallback")


def get_device(n_qubits: int = N_QUBITS, device_name: str = QUANTUM_DEVICE):
    """Initialize a PennyLane quantum device."""
    if not PENNYLANE_AVAILABLE:
        return None
    return qml.device(device_name, wires=n_qubits)


# ─── Encoding Layers ─────────────────────────────────────────────────────────
def angle_encoding(x, n_qubits: int = N_QUBITS):
    """
    Angle encoding: x[i] → RY(x[i]) on qubit i
    
    Encodes n_qubits classical features as rotation angles.
    The feature vector must have length exactly n_qubits.
    
    |0⟩ ──RY(x₀)──
    |0⟩ ──RY(x₁)──
    ...
    """
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)


def amplitude_encoding(x, n_qubits: int = N_QUBITS):
    """
    Amplitude encoding: encodes normalized feature vector as state amplitudes.
    
    Encodes 2^n_qubits values into one quantum state.
    Requires ||x|| = 1 (normalized input).
    """
    dim = 2 ** n_qubits
    # Pad or truncate to exactly 2^n_qubits
    if len(x) < dim:
        padded = np.zeros(dim)
        padded[:len(x)] = x
    else:
        padded = x[:dim]

    # Normalize
    norm = np.linalg.norm(padded)
    if norm > 1e-8:
        padded = padded / norm
    else:
        padded[0] = 1.0

    qml.AmplitudeEmbedding(padded, wires=range(n_qubits), normalize=False)


def iqp_encoding(x, n_qubits: int = N_QUBITS):
    """
    IQP (Instantaneous Quantum Polynomial) encoding.
    Applies Hadamard then RZ(x[i]) then entangling ZZ interactions.
    Creates richer feature maps.
    """
    # Hadamard layer
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    # RZ encoding
    for i in range(n_qubits):
        qml.RZ(x[i], wires=i)

    # ZZ interactions (second-order terms)
    for i in range(n_qubits - 1):
        qml.IsingZZ(x[i] * x[i + 1], wires=[i, i + 1])


# ─── Variational Ansatz ───────────────────────────────────────────────────────
def strongly_entangling_ansatz(params, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
    """
    Strongly Entangling Layers ansatz.
    
    Each layer:
      1. Rot(phi, theta, omega) on each qubit (general single-qubit rotation)
      2. CNOT ladder: qubit i → qubit (i+1) % n_qubits
    
    Parameters: params.shape = (n_layers, n_qubits, 3)
    """
    for layer in range(n_layers):
        # Single-qubit rotations
        for i in range(n_qubits):
            phi, theta, omega = params[layer, i]
            qml.Rot(phi, theta, omega, wires=i)

        # Entangling gates (CNOT ladder)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])


def hardware_efficient_ansatz(params, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
    """
    Hardware-efficient ansatz (simpler, good for NISQ devices).
    
    Each layer:
      1. RY on each qubit
      2. RZ on each qubit
      3. CZ between neighboring qubits
    
    Parameters: params.shape = (n_layers, n_qubits, 2) [RY + RZ]
    """
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[layer, i, 0], wires=i)
            qml.RZ(params[layer, i, 1], wires=i)

        for i in range(0, n_qubits - 1, 2):
            qml.CZ(wires=[i, i + 1])

        for i in range(1, n_qubits - 1, 2):
            qml.CZ(wires=[i, i + 1])


# ─── Full VQC Circuits ────────────────────────────────────────────────────────
def build_vqc_angle(
    n_qubits: int = N_QUBITS,
    n_layers: int = N_LAYERS,
    device_name: str = QUANTUM_DEVICE,
) -> Tuple[Callable, np.ndarray]:
    """
    Build a VQC with angle encoding and strongly entangling ansatz.
    
    Returns:
        circuit: qml.QNode callable
        init_params: initial random parameters
    """
    if not PENNYLANE_AVAILABLE:
        return _fallback_circuit, np.random.randn(n_layers, n_qubits, 3)

    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, params):
        """
        Full circuit: encode → ansatz → measure
        
        Args:
            x: input features (n_qubits,)
            params: variational parameters (n_layers, n_qubits, 3)
            
        Returns:
            list of PauliZ expectation values
        """
        # Encoding layer
        angle_encoding(x, n_qubits)

        # Variational ansatz
        strongly_entangling_ansatz(params, n_qubits, n_layers)

        # Measurements: ⟨Z⟩ on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    init_params = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 3))
    return circuit, init_params


def build_vqc_amplitude(
    n_qubits: int = N_QUBITS,
    n_layers: int = N_LAYERS,
    device_name: str = QUANTUM_DEVICE,
) -> Tuple[Callable, np.ndarray]:
    """
    Build a VQC with amplitude encoding.
    Input can be up to 2^n_qubits dimensional.
    """
    if not PENNYLANE_AVAILABLE:
        return _fallback_circuit, np.random.randn(n_layers, n_qubits, 3)

    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, params):
        amplitude_encoding(x, n_qubits)
        strongly_entangling_ansatz(params, n_qubits, n_layers)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    init_params = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 3))
    return circuit, init_params


def build_iqp_vqc(
    n_qubits: int = N_QUBITS,
    n_layers: int = N_LAYERS,
    device_name: str = QUANTUM_DEVICE,
) -> Tuple[Callable, np.ndarray]:
    """VQC with IQP encoding (richer feature map)."""
    if not PENNYLANE_AVAILABLE:
        return _fallback_circuit, np.random.randn(n_layers, n_qubits, 2)

    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, params):
        iqp_encoding(x, n_qubits)
        hardware_efficient_ansatz(params, n_qubits, n_layers)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    init_params = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 2))
    return circuit, init_params


def _fallback_circuit(x, params):
    """Fallback when PennyLane is unavailable — classical approximation."""
    result = np.tanh(x @ params[0, :, 0])
    return list(result)


def get_param_count(n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS) -> Dict:
    """Return parameter counts for different circuit types."""
    return {
        "angle_vqc": n_layers * n_qubits * 3,
        "hw_efficient": n_layers * n_qubits * 2,
        "iqp_vqc": n_layers * n_qubits * 2,
    }


def draw_circuit(encoding: str = "angle"):
    """Print ASCII circuit diagram."""
    if not PENNYLANE_AVAILABLE:
        print("PennyLane not available")
        return

    n_qubits = min(N_QUBITS, 4)  # Limit for display
    dev = qml.device("default.qubit", wires=n_qubits)

    if encoding == "angle":
        @qml.qnode(dev)
        def circuit(x, params):
            angle_encoding(x, n_qubits)
            strongly_entangling_ansatz(params, n_qubits, min(N_LAYERS, 2))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        x = np.random.randn(n_qubits)
        params = np.random.randn(2, n_qubits, 3)
        print(qml.draw(circuit)(x, params))

    elif encoding == "amplitude":
        @qml.qnode(dev)
        def circuit(x, params):
            amplitude_encoding(x, n_qubits)
            strongly_entangling_ansatz(params, n_qubits, min(N_LAYERS, 2))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        x = np.random.randn(2**n_qubits)
        params = np.random.randn(2, n_qubits, 3)
        print(qml.draw(circuit)(x, params))


# Fix missing Dict import
from typing import Dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"PennyLane available: {PENNYLANE_AVAILABLE}")
    print(f"\nParameter counts:")
    for name, count in get_param_count().items():
        print(f"  {name}: {count} parameters")

    if PENNYLANE_AVAILABLE:
        print("\n─── Angle Encoding Circuit ───")
        draw_circuit("angle")
        print("\n─── Testing VQC ───")
        circuit, params = build_vqc_angle(n_qubits=4, n_layers=2)
        x = np.random.randn(4)
        out = circuit(x, params)
        print(f"Input shape: {x.shape}")
        print(f"Output (⟨Z⟩): {out}")
