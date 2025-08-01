import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from new_idea.Only_GRU_model import Only_GRU, LinearLogits, LinearUnit

class AdapterGRU(nn.Module):
    def __init__(self, type_dim, geo_dim, time_dim, hidden_dim, num_classes, num_layers=1):
        super().__init__()
        self.type_dim = type_dim  # POI type encoding dimension
        self.geo_dim = geo_dim  # Geographic feature dimension
        self.time_dim = time_dim  # Time encoding dimension

        # Geographic feature adapter (key for transfer)
        self.geo_adapter = nn.Sequential(
            nn.Linear(geo_dim, geo_dim),
            nn.LayerNorm(geo_dim),
            nn.GELU(),
            nn.Linear(geo_dim, geo_dim)
        )

        # Time feature adapter (key for transfer)
        self.time_adapter = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.LayerNorm(time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # GRU layer (shared core knowledge, most parameters frozen)
        self.encoder = nn.GRU(
            input_size=type_dim + geo_dim + time_dim,  # Adapted feature input
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.logits = LinearLogits(hidden_dim, hidden_dim, adapter_dim, num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_len, type_dim+geo_dim+time_dim]
        """
        # Separate features
        type_features = x[:, :, :self.type_dim]  # POI type features
        geo_features = x[:, :, self.type_dim:self.type_dim + self.geo_dim]  # Geographic features
        time_features = x[:, :, self.type_dim + self.geo_dim:]  # Time features

        # Adapt spatio-temporal features
        adapted_geo = self.geo_adapter(geo_features)
        adapted_time = self.time_adapter(time_features)

        # Recombine features and input to GRU encoder
        adapted_input = torch.cat([type_features, adapted_geo, adapted_time], dim=-1)
        gru_output, _ = self.encoder(adapted_input)

        # Get last step hidden state (use [:,-1,:] for multiple GRU layers)
        hidden = gru_output[:, -1, :]  # [B, hidden_dim]
        return hidden


class AdapterTUL(nn.Module):
    def __init__(self, type_dim, geo_dim, time_dim, hidden_dim, num_classes, dropout_prob, num_layers, h_dim, z_dim, device):
        super().__init__()
        self.type_dim = type_dim  # POI type encoding dimension
        self.geo_dim = geo_dim  # Geographic feature dimension
        self.time_dim = time_dim  # Time encoding dimension

        # Geographic feature adapter (key for transfer)
        self.geo_adapter = nn.Sequential(
            nn.Linear(geo_dim, geo_dim),
            nn.LayerNorm(geo_dim),
            nn.GELU(),
            nn.Linear(geo_dim, geo_dim)
        )

        # Time feature adapter (key for transfer)
        self.time_adapter = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.LayerNorm(time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # GRU layer (shared core knowledge, most parameters frozen)
        self.encoder = Only_GRU(type_dim+geo_dim+time_dim, hidden_dim, hidden_dim, dropout_prob, num_layers, device)

        # Reparameterization layers
        self.mean = LinearUnit(h_dim, z_dim // 4, False)
        self.log_var = nn.Linear(h_dim, z_dim // 4)

        self.mean_man = nn.Linear(h_dim, z_dim // 4)
        self.log_var_man = nn.Linear(h_dim, 1)
        self.z_mlp = nn.Linear(h_dim // 4, h_dim)

        # Autoencoder for reconstruction error measurement
        self.autoencoder = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, h_dim),
        )
        self.auto_mes = nn.MSELoss(reduction='mean')
        self.tanh = nn.Tanh()

        # Classifier (re-initialized)
        self.logistic = LinearLogits(hidden_dim, hidden_dim, dropout_prob, num_classes, device)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        z = (mu + eps * std)
        return z

    def forward(self, input, user_label, lengths, mask):
        """
        input: [batch_size, seq_len, type_dim+geo_dim+time_dim]
        """
        # GRU encoding
        # Separate features
        type_features = input[:, :, :self.type_dim]  # type
        geo_features = input[:, :, self.type_dim:self.type_dim + self.geo_dim]  # geo
        time_features = input[:, :, self.type_dim + self.geo_dim:]  # time

        # Adapt spatio-temporal features
        adapted_geo = self.geo_adapter(geo_features)
        adapted_time = self.time_adapter(time_features)

        # Recombine features and input to GRU
        adapted_input = torch.cat([type_features, adapted_geo, adapted_time], dim=-1)
        gru_output = self.encoder(adapted_input, lengths, mask)

        encode_hidden = gru_output

        # Extract information from encoder hidden state
        h = encode_hidden
        # Extract mu and log_var from hidden state h, measure reconstruction error h_mse and var (KL divergence)
        mu = self.mean(h)
        mu = mu / mu.norm(dim=-1, keepdim=True)
        var = F.softplus(self.log_var_man(h)) + 1
        z = self.reparameterize(mu, var)  # Reparameterize Gaussian distribution parameters to generate latent variable z
        h_enc = self.autoencoder(h)
        h_mse = self.auto_mes(h_enc, h)

        # Logistic classification
        logistic_output, logistic_preds, logistic_loss = self.logistic(encode_hidden, user_label)

        return h, h_mse, mu, var, logistic_output, logistic_preds, logistic_loss