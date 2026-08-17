from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MultiModalCVAECNNConfig:
    modalities: List[str]
    signal_modalities: Optional[List[str]] = None  # default: ['filtered','powerspec']
    scalar_modalities: Optional[List[str]] = None  # default: ['att','med']
    dropout: float = 0.5
    cvae_latent_dim: int = 64
    use_cvae: bool = True
    total_features: Optional[int] = None


class MultiModalCVAECNN(nn.Module):
    """Modularized MultiModalCVAECNN (from your train_cnn.py).

    Expects each modality tensor shaped (B, T, F). (Your original was (B,10,4)).
    """

    def __init__(self, num_classes: int, cfg: MultiModalCVAECNNConfig, cvae_model: Optional[nn.Module] = None):
        super().__init__()
        self.num_classes = num_classes
        self.cfg = cfg

        self.signal_modalities = cfg.signal_modalities or ["filtered", "powerspec"]
        self.scalar_modalities = cfg.scalar_modalities or ["att", "med"]

        self.use_cvae = bool(cfg.use_cvae and cvae_model is not None)
        self.cvae = cvae_model if self.use_cvae else None
        if self.use_cvae:
            self.cvae.eval()
            for p in self.cvae.parameters():
                p.requires_grad = False

        self.signal_branch = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.scalar_branch = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        cnn_out = 2 * 64 + 2 * 16  # 160
        clf_in = cnn_out + (cfg.cvae_latent_dim if self.use_cvae else 0)

        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout * 0.8),
            nn.Linear(64, num_classes),
        )

    def _cvae_features(self, x_dict: Dict[str, torch.Tensor], labels: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.use_cvae or self.cvae is None:
            b = next(iter(x_dict.values())).shape[0]
            return torch.zeros(b, self.cfg.cvae_latent_dim, device=next(self.parameters()).device)

        flats = []
        for m in self.cfg.modalities:
            flats.append(x_dict[m].reshape(x_dict[m].shape[0], -1))
        x_flat = torch.cat(flats, dim=1)

        if labels is None:
            labels = torch.randint(0, self.num_classes, (x_flat.shape[0],), device=x_flat.device)
        labels = labels.long()
        mu, _ = self.cvae.encode(x_flat, labels)
        return mu

    def forward(self, x_dict: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None):
        sig_feats = [self.signal_branch(x_dict[m].transpose(1, 2)) for m in self.signal_modalities]
        sca_feats = [self.scalar_branch(x_dict[m].transpose(1, 2)) for m in self.scalar_modalities]
        cnn_feat = torch.cat(sig_feats + sca_feats, dim=1)

        if self.use_cvae:
            cvae_feat = self._cvae_features(x_dict, labels)
            feat = torch.cat([cnn_feat, cvae_feat], dim=1)
        else:
            feat = cnn_feat
        return self.classifier(feat)
