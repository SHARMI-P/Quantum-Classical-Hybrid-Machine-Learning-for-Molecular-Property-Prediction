# models/hybrid_model.py — Hybrid Quantum-Classical Model
"""
Combines a quantum feature extractor with classical regression head.

Two hybrid architectures:

1. QFM + Classical (Quantum Feature Map + Ridge/RF):
   X → [Quantum Circuit] → Q_features → [Classical Regressor] → y
   - Trains quantum params and classical regressor jointly (or separately)
   - Fast to train, deterministic

2. Hybrid Neural Net (PyTorch):
   X → [Classical encoder] → [Quantum Layer] → [Classical decoder] → y
   - End-to-end differentiable via parameter-shift rule
   - Slower but more expressive

The key insight: quantum circuits can create feature maps in exponentially 
large Hilbert spaces, potentially separating data that is hard to separate 
classically.
"""

import os
import sys
import time
import logging
import numpy as np
import pickle
from typing import Dict, Any, Optional, List, Tuple

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_QUBITS, N_LAYERS, HYBRID_LR, HYBRID_EPOCHS,
                    HYBRID_BATCH_SIZE, HIDDEN_DIM, RANDOM_STATE, ENCODING)

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


# ─── Architecture 1: QFM + Classical Regressor ───────────────────────────────
class QuantumClassicalHybrid:
    """
    Quantum Feature Map (QFM) followed by a classical regressor.
    
    Pipeline:
        X_pca (n_qubits) → VQC → Q_features (n_qubits) → Ridge/RF → y
    
    The quantum circuit transforms features into a richer representation
    via entanglement and superposition, which the classical model then fits.
    
    This is the "kernel trick" approach to quantum ML.
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        encoding: str = ENCODING,
        classical_head: str = "ridge",
        optimize_circuit: bool = True,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.classical_head = classical_head
        self.optimize_circuit = optimize_circuit
        self.is_fitted = False
        self.train_time = None
        self.train_losses = []

        # Quantum feature extractor
        from quantum.quantum_layer import QuantumFeatureExtractor
        self.qfe = QuantumFeatureExtractor(n_qubits, n_layers, encoding)

        # Classical head
        if classical_head == "ridge":
            self.regressor = Ridge(alpha=1.0)
        elif classical_head == "rf":
            self.regressor = RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            )
        else:
            self.regressor = Ridge(alpha=1.0)

    @property
    def name(self):
        return f"Hybrid VQC ({self.encoding} encoding)"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_optimization_steps: int = 30,
    ) -> "QuantumClassicalHybrid":
        """
        Train the hybrid model:
        1. Optionally optimize circuit parameters on training data
        2. Extract quantum features from all training samples
        3. Train classical head on quantum features
        """
        t0 = time.time()
        logger.info(f"\nTraining {self.name}...")
        logger.info(f"  Input shape: {X_train.shape}")

        # Step 1: Optimize quantum circuit parameters (optional)
        if self.optimize_circuit and PENNYLANE_AVAILABLE:
            logger.info("  Optimizing quantum circuit parameters...")
            losses = self.qfe.optimize_params(X_train, y_train, n_steps=n_optimization_steps)
            self.train_losses = losses

        # Step 2: Extract quantum features
        logger.info("  Extracting quantum features...")
        Q_train = self.qfe.transform(X_train)
        logger.info(f"  Quantum features shape: {Q_train.shape}")

        # Step 3: Train classical head
        logger.info(f"  Training {self.classical_head} regressor on quantum features...")
        self.regressor.fit(Q_train, y_train)

        self.train_time = time.time() - t0
        self.is_fitted = True
        logger.info(f"  Total training time: {self.train_time:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted
        Q = self.qfe.transform(X)
        return self.regressor.predict(Q)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        return {
            "mae": mean_absolute_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "r2": r2_score(y, y_pred),
            "train_time": self.train_time or 0.0,
        }

    def get_complexity(self) -> Dict[str, Any]:
        n_circuit_params = self.n_layers * self.n_qubits * 3
        return {
            "n_circuit_params": n_circuit_params,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "encoding": self.encoding,
            "model_type": "hybrid_quantum_classical",
        }


# ─── Architecture 2: Hybrid Neural Network (PyTorch) ─────────────────────────
class HybridNeuralNet(nn.Module if TORCH_AVAILABLE else object):
    """
    End-to-end differentiable hybrid quantum-classical neural network.
    
    Architecture:
        Input (n_features) 
        → Classical Encoder (Linear→ReLU→Linear, to n_qubits)
        → Quantum Layer (VQC, outputs n_qubits ⟨Z⟩ values)
        → Classical Decoder (Linear→ReLU→Linear→scalar)
        → Output (scalar property)
    
    Trained end-to-end with MSE loss via Adam optimizer.
    Gradients through quantum layer via parameter-shift rule.
    """

    def __init__(
        self,
        input_dim: int = N_QUBITS,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for HybridNeuralNet")
        super().__init__()

        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical encoder: input → n_qubits
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh(),  # Ensures output in [-1, 1], scaled to [-π, π] below
        )

        # Quantum layer
        if PENNYLANE_AVAILABLE:
            from quantum.quantum_layer import QuantumLayer
            self.quantum_layer = QuantumLayer(n_qubits, n_layers)
        else:
            # Classical simulation of quantum layer
            self.quantum_layer = nn.Sequential(
                nn.Linear(n_qubits, n_qubits * 2),
                nn.Tanh(),
                nn.Linear(n_qubits * 2, n_qubits),
                nn.Tanh(),
            )

        # Classical decoder: n_qubits → scalar
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # Encode to qubit-compatible dimension
        encoded = self.encoder(x) * np.pi  # Scale to [-π, π]

        # Quantum transformation
        quantum_features = self.quantum_layer(encoded)

        # Decode to scalar
        output = self.decoder(quantum_features)
        return output.squeeze(-1)


class HybridNeuralNetWrapper:
    """
    Training wrapper for HybridNeuralNet with sklearn-like interface.
    Handles training loop, learning rate scheduling, early stopping.
    """

    def __init__(
        self,
        input_dim: int = N_QUBITS,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        lr: float = HYBRID_LR,
        epochs: int = HYBRID_EPOCHS,
        batch_size: int = HYBRID_BATCH_SIZE,
    ):
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_losses = []
        self.val_losses = []
        self.train_time = None
        self.is_fitted = False
        self.name = "Hybrid Neural Network (VQC)"

        if TORCH_AVAILABLE:
            self.model = HybridNeuralNet(input_dim, n_qubits, n_layers, hidden_dim)
            self.device = torch.device("cpu")
            self.model.to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "HybridNeuralNetWrapper":
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable — skipping HybridNeuralNet")
            return self

        t0 = time.time()
        logger.info(f"\nTraining {self.name}...")

        # Convert to tensors
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            epoch_loss /= len(X_train)
            self.train_losses.append(epoch_loss)
            scheduler.step()

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(X_val).to(self.device)
                    y_v = torch.FloatTensor(y_val).to(self.device)
                    val_pred = self.model(X_v)
                    val_loss = loss_fn(val_pred, y_v).item()
                self.val_losses.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 5 == 0:
                val_str = f", val_loss={val_loss:.4f}" if X_val is not None else ""
                logger.info(f"  Epoch {epoch+1:3d}/{self.epochs}: loss={epoch_loss:.4f}{val_str}")

        # Restore best model
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

        self.train_time = time.time() - t0
        self.is_fitted = True
        logger.info(f"  Training complete in {self.train_time:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted and TORCH_AVAILABLE
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            y_pred = self.model(X_t)
        return y_pred.cpu().numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        return {
            "mae": mean_absolute_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "r2": r2_score(y, y_pred),
            "train_time": self.train_time or 0.0,
        }

    def get_complexity(self) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            return {}
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "model_type": "hybrid_neural_network",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    n_samples, n_qubits = 100, 4
    X = np.random.randn(n_samples, n_qubits).astype(np.float32)
    y = np.sum(X ** 2, axis=1) + np.random.randn(n_samples) * 0.1

    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    print("\n─── Testing QuantumClassicalHybrid ───")
    qch = QuantumClassicalHybrid(n_qubits=n_qubits, n_layers=1, encoding="angle",
                                  optimize_circuit=False)
    qch.fit(X_train, y_train, n_optimization_steps=5)
    metrics = qch.evaluate(X_test, y_test)
    print(f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")

    if TORCH_AVAILABLE:
        print("\n─── Testing HybridNeuralNet ───")
        hnn = HybridNeuralNetWrapper(input_dim=n_qubits, n_qubits=n_qubits,
                                      n_layers=1, epochs=10)
        hnn.fit(X_train, y_train, X_test, y_test)
        metrics = hnn.evaluate(X_test, y_test)
        print(f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
        print(f"Complexity: {hnn.get_complexity()}")
