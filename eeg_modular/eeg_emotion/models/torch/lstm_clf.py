import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 hidden: int = 128, num_layers: int = 2, dropout: float = 0.35,
                 pooling: str = "avgmax"):
        super().__init__()
        self.pooling = pooling.lower()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        feat_dim = hidden * 2
        if self.pooling == "avgmax":
            head_in = feat_dim * 2
        else:
            head_in = feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)  # [B, T, 2H]
        if self.pooling == "last":
            feat = out[:, -1, :]
        elif self.pooling == "avg":
            feat = out.mean(dim=1)
        elif self.pooling == "max":
            feat = out.max(dim=1).values
        elif self.pooling == "avgmax":
            feat = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        else:
            feat = out.mean(dim=1)
        return self.head(feat)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.net(x)
