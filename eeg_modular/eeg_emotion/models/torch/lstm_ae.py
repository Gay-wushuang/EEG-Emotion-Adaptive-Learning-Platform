import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.25, bidir_decoder: bool = True):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        dec_hidden = hidden_dim
        self.from_latent = nn.Linear(latent_dim, dec_hidden)

        self.decoder = nn.LSTM(
            input_size=dec_hidden,
            hidden_size=dec_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bool(bidir_decoder),
        )
        out_dim = dec_hidden * (2 if bidir_decoder else 1)
        self.to_recon = nn.Linear(out_dim, input_dim)

    def encode(self, x):
        # x: [B, T, F]
        out, _ = self.encoder(x)       # out: [B, T, H]
        h_last = out[:, -1, :]         # [B, H]
        z = self.to_latent(h_last)     # [B, Z]
        return z

    def forward(self, x):
        z = self.encode(x)                         # [B, Z]
        h0 = self.from_latent(z).unsqueeze(1)      # [B, 1, H]
        dec_in = h0.repeat(1, x.size(1), 1)        # [B, T, H]
        dec_out, _ = self.decoder(dec_in)          # [B, T, H*dir]
        recon = self.to_recon(dec_out)             # [B, T, F]
        return recon, z
