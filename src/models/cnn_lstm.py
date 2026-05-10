"""
CNN-LSTM Hybrid Model for Intrusion Detection
=============================================
Combines CNN for spatial feature extraction with LSTM for temporal 
pattern recognition in network traffic flows.

Architecture:
    Input (N features) → Reshape to (1, N) 
    → Conv1D(64) → BatchNorm → ReLU → Dropout
    → Conv1D(128) → BatchNorm → ReLU → Dropout
    → LSTM(128, 2 layers) → Dropout
    → FC(64) → ReLU → Dropout → FC(num_classes)

Usage:
    from src.models.cnn_lstm import CNNLSTM
    model = CNNLSTM(input_dim=64, num_classes=2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid model for network intrusion detection.

    CNN layers extract local spatial patterns from network flow features.
    LSTM layers capture temporal dependencies across feature sequences.
    Combined architecture achieves high accuracy on CICIDS2017 benchmark.

    Args:
        input_dim: Number of input features per sample
        num_classes: Number of output classes (2 for binary, 8 for multi-class)
        cnn_filters: List of filter sizes for CNN layers
        kernel_size: Convolution kernel size
        lstm_hidden: LSTM hidden state size
        lstm_layers: Number of LSTM layers
        fc_hidden: Fully connected hidden layer size
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 64,
        num_classes: int = 2,
        cnn_filters: list = None,
        kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        fc_hidden: int = 64,
        dropout: float = 0.3,
    ):
        super(CNNLSTM, self).__init__()

        if cnn_filters is None:
            cnn_filters = [64, 128]

        self.input_dim = input_dim
        self.num_classes = num_classes

        # ---- CNN Layers ----
        # Input shape: (batch, 1, input_dim) — 1 channel
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=cnn_filters[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # same padding
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters[0])

        self.conv2 = nn.Conv1d(
            in_channels=cnn_filters[0],
            out_channels=cnn_filters[1],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(cnn_filters[1])

        self.cnn_dropout = nn.Dropout(dropout)

        # ---- LSTM Layers ----
        # Input: (batch, seq_len, features) — seq_len = input_dim, features = cnn_filters[-1]
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # ---- Classifier Head ----
        self.fc1 = nn.Linear(lstm_hidden, fc_hidden)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Reshape: (batch, input_dim) → (batch, 1, input_dim) for Conv1D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))   # (batch, 64, input_dim)
        x = self.cnn_dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))   # (batch, 128, input_dim)
        x = self.cnn_dropout(x)

        # Reshape for LSTM: (batch, channels, seq_len) → (batch, seq_len, channels)
        x = x.permute(0, 2, 1)  # (batch, input_dim, 128)

        # LSTM temporal processing
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden)

        # Use last hidden state
        x = lstm_out[:, -1, :]  # (batch, hidden)
        x = self.lstm_dropout(x)

        # Classifier
        x = F.relu(self.fc1(x))  # (batch, fc_hidden)
        x = self.fc_dropout(x)
        logits = self.fc2(x)     # (batch, num_classes)

        return logits

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Extract model gradients for federated learning.
        Returns dict of parameter_name -> gradient tensor.
        """
        gradients = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients

    def set_parameters(self, state_dict: Dict[str, torch.Tensor]):
        """Load parameters from a state dict (for federated aggregation)."""
        self.load_state_dict(state_dict)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters as a dict."""
        return {name: param.data.clone() for name, param in self.named_parameters()}

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineAutoencoder(nn.Module):
    """
    Autoencoder baseline for anomaly detection.
    Trains on benign traffic only; high reconstruction error = anomaly.
    """

    def __init__(self, input_dim: int = 64, encoding_dim: int = 16, dropout: float = 0.2):
        super(BaselineAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, encoding_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error."""
        reconstructed = self.forward(x)
        error = F.mse_loss(reconstructed, x, reduction="none").mean(dim=1)
        return error


def build_model(
    model_type: str = "cnn_lstm",
    input_dim: int = 64,
    num_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """Factory function to build models."""
    if model_type == "cnn_lstm":
        return CNNLSTM(input_dim=input_dim, num_classes=num_classes, **kwargs)
    elif model_type == "autoencoder":
        return BaselineAutoencoder(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Quick test
    model = CNNLSTM(input_dim=64, num_classes=2)
    print(f"CNN-LSTM Model: {model.count_parameters():,} parameters")
    print(model)

    # Test forward pass
    x = torch.randn(32, 64)  # batch of 32, 64 features
    out = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output sample: {F.softmax(out[0], dim=0).detach().numpy()}")
