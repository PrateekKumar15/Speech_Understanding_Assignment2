from __future__ import annotations

import torch
import torch.nn as nn


class FrameLIDNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        hidden_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc, _ = self.encoder(x)
        return self.classifier(enc)
