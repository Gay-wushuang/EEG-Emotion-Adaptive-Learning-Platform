import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim=64, hidden_dim=256):
        """
        参数顺序: input_dim, num_classes, latent_dim
        """
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embedding_dim = 16

        # 标签嵌入
        self.label_emb = nn.Embedding(num_classes, self.embedding_dim)

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + self.embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # 解码器
        decoder_input_dim = latent_dim + self.embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, input_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x, y):
        y_emb = self.label_emb(y)
        h = torch.cat([x, y_emb], dim=1)
        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_emb = self.label_emb(y)
        h = torch.cat([z, y_emb], dim=1)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=0.1):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def generate(self, y, num_samples=None):
        self.eval()
        if num_samples is None:
            num_samples = len(y)
        
        device = next(self.parameters()).device
        y = y.to(device) if torch.is_tensor(y) else torch.tensor(y, dtype=torch.long, device=device)
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            x_gen = self.decode(z, y)
        return x_gen.cpu().numpy()
    