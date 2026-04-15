"""
quantum/circuit.py
==================
Parameterized quantum circuits using PennyLane for hybrid ML.

Two encoding strategies:
  - AngleEncoding  : RY rotations with one qubit per feature
  - AmplitudeEncoding : amplitude embedding (requires 2^n_qubits features)

Variational ansatz: strongly entangling layers (SEL).

The circuit outputs expectation values <Z_i> for each qubit,
which become the quantum feature vector fed into a classical head.
"""

import numpy as np

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("[quantum] PennyLane not found – using SimulatedQuantum fallback.")


# ── Simulated quantum feature map (no PennyLane needed) ──────────────────────
class SimulatedQuantumFeatureMap:
    """
    CPU-based simulation that approximates quantum-like feature transformations.

    Applies random Fourier features with trigonometric combinations to mimic
    the non-linear feature map a real quantum circuit would produce.

    This is used as a drop-in replacement when PennyLane is unavailable,
    ensuring the hybrid pipeline still runs end-to-end.
    """

    def __init__(self, n_qubits=4, n_layers=2, random_state=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        rng = np.random.default_rng(random_state)
        # Random projections W: (n_qubits, n_qubits * n_layers)
        self.W = rng.standard_normal((n_qubits, n_qubits * n_layers)) * 0.5
        self.b = rng.uniform(0, 2 * np.pi, n_qubits * n_layers)
        # Learnable (for gradient-free optimisation in the hybrid model)
        self.params = rng.uniform(0, np.pi, (n_layers, n_qubits, 3))

    def transform(self, X):
        """
        Map X (n_samples, n_qubits) → quantum_features (n_samples, n_qubits).
        """
        # Angle-encode via RY-like non-linearity
        Z = np.cos(X @ self.W[:, :self.n_qubits] + self.b[:self.n_qubits])
        for l in range(self.n_layers):
            # Simulated entanglement: circular CNOT-like mixing
            Z = np.roll(Z, 1, axis=1) * np.sin(Z) + np.cos(Z)
            # Variational rotation
            Z = np.tanh(Z @ np.diag(np.cos(self.params[l, :, 0])))
        # Outputs in [-1, 1] mimic <Z> expectation values
        return np.tanh(Z)

    @property
    def output_dim(self):
        return self.n_qubits

    def circuit_diagram(self):
        lines = [
            "SimulatedQuantumFeatureMap (PennyLane not installed)",
            f"  Qubits  : {self.n_qubits}",
            f"  Layers  : {self.n_layers}",
            "  Encoding: Angle (RY-approximation)",
            "  Ansatz  : Entangling + variational rotations",
            "  Output  : tanh(<Z_i>) expectation values",
        ]
        return "\n".join(lines)


# ── Real PennyLane circuit ────────────────────────────────────────────────────
class PennyLaneQuantumCircuit:
    """
    Real variational quantum circuit using PennyLane.

    Architecture:
      1. Angle encoding: RY(x_i) on each qubit
      2. Strongly entangling layers (SEL) with trainable parameters
      3. Measure <PauliZ> on each qubit
    """

    def __init__(self, n_qubits=4, n_layers=2, random_state=42):
        assert PENNYLANE_AVAILABLE, "PennyLane required."
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        rng = np.random.default_rng(random_state)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.params = rng.uniform(0, 2 * np.pi, shape)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="numpy")
        def circuit(inputs, weights):
            # Angle encoding
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit = circuit

    def transform(self, X):
        """
        Map X (n_samples, n_qubits) → quantum_features (n_samples, n_qubits).
        """
        out = []
        for x in X:
            out.append(self._circuit(x, self.params))
        return np.array(out)

    @property
    def output_dim(self):
        return self.n_qubits

    def circuit_diagram(self):
        dummy_x = np.zeros(self.n_qubits)
        return qml.draw(self._circuit)(dummy_x, self.params)


# ── Public factory ────────────────────────────────────────────────────────────
class QuantumFeatureMap:
    """
    Factory that returns a PennyLane circuit if available, otherwise the
    simulated fallback. Both expose the same .transform() interface.
    """

    def __new__(cls, n_qubits=4, n_layers=2, random_state=42):
        if PENNYLANE_AVAILABLE:
            print(f"[quantum] Using PennyLane VQC  ({n_qubits} qubits, {n_layers} layers)")
            return PennyLaneQuantumCircuit(n_qubits, n_layers, random_state)
        else:
            print(f"[quantum] Using SimulatedQFM   ({n_qubits} qubits, {n_layers} layers)")
            return SimulatedQuantumFeatureMap(n_qubits, n_layers, random_state)
