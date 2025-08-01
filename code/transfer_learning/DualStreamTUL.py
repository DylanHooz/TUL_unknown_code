import torch
import torch.nn.functional as F
from torch import nn
import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Only_GRU_model import LinearLogits, Only_GRU
from GRL import GRL_Layer

class GeoTimeAdapter(nn.Module):
    def __init__(self, geo_dim, time_dim):
        super().__init__()
        self.geo_adapter = nn.Sequential(
            nn.Linear(geo_dim, 64),
            nn.ReLU(),
            nn.Linear(64, geo_dim)
        )
        self.time_adapter = nn.Sequential(
            nn.Linear(time_dim, 64),
            nn.ReLU(),
            nn.Linear(64, time_dim)
        )

    def forward(self, geo_features, time_features):
        """Adapt source domain geographic and time features to target domain space"""
        adapted_geo = self.geo_adapter(geo_features)
        adapted_time = self.time_adapter(time_features)
        return adapted_geo, adapted_time


class TwoStageDualStreamModel(nn.Module):
    def __init__(self, type_dim, geo_dim, time_dim, hidden_dim, dropout_prob, num_layers, source_num_classes, target_num_classes, device):
        super().__init__()
        self.type_dim = type_dim
        self.geo_dim = geo_dim
        self.time_dim = time_dim
        self.total_dim = type_dim + geo_dim + time_dim

        # Time and geo adapters (only for source domain)
        self.adapter = GeoTimeAdapter(geo_dim, time_dim)

        # Shared GRU layer
        self.encoder_gru = Only_GRU(self.total_dim, hidden_dim, hidden_dim, dropout_prob, num_layers,device)
        # For pretraining trajectory reconstruction
        self.decoder_gru = nn.GRU(self.total_dim, hidden_dim, num_layers, batch_first=True)
        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.total_dim)
        )

        # Source TUL classifier (default: source and target domains have same number of classes but different IDs)
        self.source_tul_classifier = LinearLogits(hidden_dim, hidden_dim, dropout_prob, source_num_classes, device)
        # Target TUL classifier
        self.tul_classifier = LinearLogits(hidden_dim, hidden_dim, dropout_prob, target_num_classes, device)

        # Domain discriminator (for adversarial training: learn common invariant knowledge between different domains)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

        # Gradient reversal layer
        self.grl = GRL_Layer()

    def forward(self, source_trajs, source_labels, source_length, source_mask,
                target_trajs, target_labels, target_length, target_mask, entropy_lambad=0.1,
                mode='pretrain'):
        """
        mode: 'pretrain' or 'finetune'
        """
        if mode == 'pretrain':
            # Classifier loss and domain loss
            # Source domain
            source_features = self._extract_features(source_trajs, source_length, source_mask)
            source_output, source_preds, source_logis_loss = self.source_tul_classifier(source_features, source_labels)
            source_probs = F.softmax(source_output, dim=1)

            # Calculate entropy loss and sample entropy weights (penalize low-entropy samples for poor generalization)
            weighted_entropy_loss, source_entropy_mean_loss, max_entropy = self.compute_adaptive_entropy_loss(source_output)

            # Target domain
            target_features = self._extract_features(target_trajs, target_length, target_mask)
            target_output, target_preds, target_logis_loss = self.tul_classifier(target_features, target_labels)

            # Total loss
            loss = (1.0 * source_logis_loss) + (entropy_lambad * weighted_entropy_loss) + (1.0 * target_logis_loss)

            return loss

        elif mode == 'finetune':
            # Second stage: only process target domain data for TUL classification
            features = self._extract_features(target_trajs,target_length,target_mask)
            output, preds, loss = self.tul_classifier(features, target_labels)
            return output, preds, loss

    def compute_adaptive_entropy_loss(self, logits, eps=1e-10):
        """
        Calculate entropy loss with adaptive weights
        """
        probs = F.softmax(logits, dim=1)

        # Calculate entropy for each sample
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)

        # Calculate entropy distribution statistics
        mean_entropy = torch.mean(entropy)
        max_entropy = torch.log(torch.tensor(logits.size(1), dtype=torch.float32, device=logits.device))

        # Normalize entropy: scale entropy values to [0,1] range
        normalized_entropy = entropy / max_entropy

        adaptive_weight = normalized_entropy

        # Weighted entropy loss: high-entropy samples contribute more to loss
        weighted_entropy_loss = torch.mean(adaptive_weight * entropy)

        return weighted_entropy_loss, mean_entropy, max_entropy

    def _extract_features(self, trajs, lengths, mask):
        """Extract GRU-encoded features (for TUL classification)"""
        # Get features directly through GRU
        output, last_hidden = self.encoder_gru(trajs, lengths, mask)
        # Use last time step hidden state
        return last_hidden.squeeze(0)

    def compute_reconstruct_loss(self, source_reconstructed, source_traj, target_reconstructed, target_traj):
        """Calculate vector prediction loss (MSE)"""
        source_loss = F.mse_loss(source_reconstructed, source_traj)
        target_loss = F.mse_loss(target_reconstructed, target_traj)
        return (source_loss + target_loss)