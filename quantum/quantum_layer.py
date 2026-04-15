# quantum/quantum_layer.py — PyTorch-compatible Quantum Layer
"""
Wraps a PennyLane quantum circuit as a PyTorch nn.Module layer.

This enables:
  - Gradient computation through the quantum circuit (parameter-shift rule)
  - Integration with standard PyTorch training loops
  - Batch processing of molecules

The quantum layer acts as a feature map:
  classical features (8-dim) → quantum circuit → quantum features (8-dim)

These quantum features are then passed to a classical head (linear layers).
"""

import os
import sys
import numpy as np
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_QUBITS, N_LAYERS, QUANTUM_DEVICE, ENCODING

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


class QuantumLayer(nn.Module if TORCH_AVAILABLE else object):
    """
    A differentiable quantum layer for use in PyTorch models.
    
    Uses PennyLane's TorchLayer to wrap a QNode as a nn.Module.
    Gradients are computed via the parameter-shift rule.
    
    Architecture:
        Input: (batch, n_qubits) tensor
        ↓
        Quantum circuit (angle encoding + variational ansatz)
        ↓
        Output: (batch, n_qubits) tensor of ⟨Z⟩ expectation values
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        encoding: str = ENCODING,
        device_name: str = QUANTUM_DEVICE,
    ):
        if TORCH_AVAILABLE:
            super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.device_name = device_name

        if PENNYLANE_AVAILABLE and TORCH_AVAILABLE:
            self._build_torch_layer()
        else:
            logger.warning("Using classical fallback for quantum layer")
            if TORCH_AVAILABLE:
                self.fallback_linear = nn.Linear(n_qubits, n_qubits)

    def _build_torch_layer(self):
        """Build PennyLane TorchLayer."""
        dev = qml.device(self.device_name, wires=self.n_qubits)

        # Define the QNode
        if self.encoding == "angle":
            @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
            def circuit(inputs, weights):
                # Angle encoding
                for i in range(self.n_qubits):
                    qml.RY(inputs[i], wires=i)
                # Variational ansatz
                for layer in range(self.n_layers):
                    for i in range(self.n_qubits):
                        qml.Rot(
                            weights[layer, i, 0],
                            weights[layer, i, 1],
                            weights[layer, i, 2],
                            wires=i
                        )
                    for i in range(self.n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        elif self.encoding == "amplitude":
            @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
            def circuit(inputs, weights):
                # Amplitude encoding
                dim = 2 ** self.n_qubits
                padded = torch.zeros(dim)
                padded[:min(len(inputs), dim)] = inputs[:min(len(inputs), dim)]
                norm = torch.norm(padded)
                if norm > 1e-8:
                    padded = padded / norm
                else:
                    padded[0] = 1.0
                qml.AmplitudeEmbedding(padded, wires=range(self.n_qubits), normalize=False)
                # Ansatz
                for layer in range(self.n_layers):
                    for i in range(self.n_qubits):
                        qml.Rot(
                            weights[layer, i, 0],
                            weights[layer, i, 1],
                            weights[layer, i, 2],
                            wires=i
                        )
                    for i in range(self.n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Wrap as TorchLayer
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        logger.info(f"Quantum layer built: {self.n_qubits} qubits, {self.n_layers} layers, {self.encoding} encoding")

    def forward(self, x):
        """
        Forward pass through quantum circuit.
        
        Args:
            x: torch.Tensor of shape (batch_size, n_qubits)
            
        Returns:
            torch.Tensor of shape (batch_size, n_qubits)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        if PENNYLANE_AVAILABLE and hasattr(self, 'qlayer'):
            # Process batch one sample at a time (quantum circuits aren't natively batched)
            outputs = []
            for i in range(x.shape[0]):
                out = self.qlayer(x[i])
                outputs.append(out)
            return torch.stack(outputs)
        else:
            # Classical fallback
            return torch.tanh(self.fallback_linear(x))

    def get_param_count(self) -> int:
        """Return total trainable parameter count."""
        if hasattr(self, 'qlayer'):
            return self.n_layers * self.n_qubits * 3
        return self.n_qubits * self.n_qubits  # fallback linear


class QuantumFeatureExtractor:
    """
    Numpy-based quantum feature extractor (no PyTorch required).
    
    Uses a fixed (pre-trained or random) quantum circuit to transform
    features. Used as a feature map for sklearn-compatible models.
    
    The circuit parameters can be optimized separately using gradient descent.
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        encoding: str = "angle",
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.params = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 3))

        if PENNYLANE_AVAILABLE:
            self._setup_circuit()
        else:
            logger.warning("PennyLane not available — using classical fallback")

    def _setup_circuit(self):
        """Set up the PennyLane circuit."""
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="autograd")
        def circuit(x, params):
            if self.encoding == "angle":
                for i in range(self.n_qubits):
                    qml.RY(x[i], wires=i)
            elif self.encoding == "amplitude":
                dim = 2 ** self.n_qubits
                padded = np.zeros(dim)
                padded[:min(len(x), dim)] = x[:min(len(x), dim)]
                norm = np.linalg.norm(padded)
                if norm > 1e-8:
                    padded = padded / norm
                else:
                    padded[0] = 1.0
                qml.AmplitudeEmbedding(padded, wires=range(self.n_qubits))

            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input features through quantum circuit.
        
        Args:
            X: np.ndarray of shape (n_samples, n_qubits)
            
        Returns:
            quantum_features: np.ndarray of shape (n_samples, n_qubits)
        """
        if not PENNYLANE_AVAILABLE:
            return np.tanh(X @ np.random.randn(self.n_qubits, self.n_qubits))

        results = []
        for i, x in enumerate(X):
            if i % 100 == 0 and i > 0:
                logger.info(f"  Quantum transform: {i}/{len(X)}")
            out = self.circuit(x, self.params)
            results.append(np.array(out))

        return np.array(results, dtype=np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """sklearn-compatible fit_transform (circuit params fixed)."""
        return self.transform(X)

    def optimize_params(self, X: np.ndarray, y: np.ndarray, n_steps: int = 50):
        """
        Optimize circuit parameters using gradient descent.
        Minimizes MSE between quantum output→linear→prediction and y.
        """
        if not PENNYLANE_AVAILABLE:
            return

        from pennylane import numpy as pnp
        from pennylane.optimize import AdamOptimizer

        params = pnp.array(self.params, requires_grad=True)
        weights = pnp.array(np.random.randn(self.n_qubits), requires_grad=True)
        opt = AdamOptimizer(stepsize=0.01)

        def cost(params, weights):
            total = 0
            for x, target in zip(X[:50], y[:50]):  # Mini-batch
                out = pnp.array(self.circuit(x, params))
                pred = pnp.dot(weights, out)
                total += (pred - target) ** 2
            return total / 50

        logger.info("Optimizing quantum circuit parameters...")
        losses = []
        for step in range(n_steps):
            params, weights = opt.step(cost, params, weights)
            if step % 10 == 0:
                loss = cost(params, weights)
                losses.append(float(loss))
                logger.info(f"  Step {step:3d}: loss = {loss:.4f}")

        self.params = np.array(params)
        self.weights = np.array(weights)
        return losses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"PennyLane: {PENNYLANE_AVAILABLE}, PyTorch: {TORCH_AVAILABLE}")

    if PENNYLANE_AVAILABLE:
        print("\n─── Testing QuantumFeatureExtractor ───")
        extractor = QuantumFeatureExtractor(n_qubits=4, n_layers=2)
        X_test = np.random.randn(5, 4)
        Q_features = extractor.transform(X_test)
        print(f"Input shape:  {X_test.shape}")
        print(f"Output shape: {Q_features.shape}")
        print(f"Output range: [{Q_features.min():.3f}, {Q_features.max():.3f}]")
        print(f"Sample output: {Q_features[0]}")
