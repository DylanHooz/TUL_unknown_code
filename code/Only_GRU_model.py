import torch
import torch.nn as nn
import torch.nn.functional as F

class Only_GRU(nn.Module):
    # Convert POI trajectories to vector representations using embedding and GRU layers
    def __init__(self,embed_size, hidden_size, output_size, dropout_prob, num_layers, device):
        super(Only_GRU, self).__init__()
        # Set input parameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.Dropout = nn.Dropout(dropout_prob)
        # Initialize GRU with batch_first=True
        self.enc_gru = nn.GRU(self.embed_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)

    def forward(self, poi_list, lengths, mask):
        '''
        :param poi_list: Embedded POI trajectory sequence [batch_size, seq_len, embed_size]
        :param lengths: Length of POI trajectories [batch_size]
        :param mask: Mask for random POI removal (1=keep, 0=mask) [batch_size, seq_len]
        :return: GRU output and last hidden state
        '''
        # Extract valid parts after masking
        mask = mask.bool()
        # Calculate actual length for each sample
        lengths = mask.sum(dim=1)

        # Create PackedSequence
        pack = nn.utils.rnn.pack_padded_sequence(
            poi_list,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        gru_output, last_hidden = self.enc_gru(pack)
        last_hidden = self.Dropout(last_hidden)
        
        return gru_output, last_hidden

class LinearLogits(nn.Module):
    '''
    Classification head after encoder for trajectory user linkage (TUL) task
    '''
    def __init__(self, input_dim, hidden_dim, dropout_prob, num_classes, device):
        super(LinearLogits, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        # Define network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)
        # Batch normalization and dropout
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        # Cross entropy loss
        self.loss = nn.CrossEntropyLoss(reduction = 'mean')

        print(self)

    def forward(self, x, user_label):
        # Forward pass
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        output = self.output(x)  # Raw logits
        # Get predictions
        preds = torch.argmax(output, dim=-1)
        # Calculate classification loss
        loss = self.loss(output, user_label)
        return output, preds, loss

class LinearUnit(nn.Module):
    # Linear unit with optional batch normalization
    def __init__(self, in_features, out_features, batch_norm=True, non_linearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        # Create linear layer with optional layer normalization
        if batch_norm is True:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features), non_linearity)

    def forward(self, x):
        return self.model(x)

class Only_GRU_TUL(nn.Module):
    # Combined seq2seq framework for trajectory user linkage
    def __init__(self, encoder, logistic, device, hidden_size, dropout_prob, h_dim, z_dim):
        super(Only_GRU_TUL, self).__init__()

        # Network models
        self.encoder = encoder
        self.logistic = logistic
        self.device = device

        self.Dropout = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(z_dim//4, hidden_size)
        self.enc_uni = nn.GRU(h_dim * 2, h_dim, batch_first=True)

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

    # Reparameterization trick for Gaussian distribution
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        z = (mu + eps * std)
        return z

    def Laplace_Distribution_reparameterize(self, mu, b):
        u = torch.rand_like(b) - 0.5
        z = mu - b * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))
        return z

    def kl_divergence_laplace(self, mu, b_q):
        mu_p = 0
        b_p = 1
        # KL divergence calculation
        kl_div = torch.log(b_p / b_q) + (torch.abs(mu - mu_p) / b_p) + (b_q / b_p) - 1
        return torch.sum(kl_div)

    # Gaussian forward pass
    def forward(self, encoder_input, decoder_input, user_label, lengths, mask):
        # Encoder forward pass
        gru_output, encode_hidden = self.encoder(encoder_input, lengths, mask)
        encode_hidden = encode_hidden.squeeze(0)
        # Extract information from encoder hidden state
        h = encode_hidden
        # Extract mu and log_var from hidden state h
        mu = self.mean(h)
        mu = mu / mu.norm(dim=-1, keepdim=True)
        var = F.softplus(self.log_var_man(h)) + 1
        z = self.reparameterize(mu, var)
        h_enc = self.autoencoder(h)
        h_mse = self.auto_mes(h_enc, h)

        # Classification
        logistic_output, logistic_preds, logistic_loss = self.logistic(encode_hidden, user_label)

        return h, h_mse, mu, var, logistic_output, logistic_preds, logistic_loss