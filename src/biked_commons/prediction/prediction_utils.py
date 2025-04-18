import torch
import dill
from torch import nn


class TorchStandardScaler(nn.Module):
    def __init__(self):
        super().__init__()
        # these will be set in .fit()
        self.register_buffer('mean', torch.tensor([]))
        self.register_buffer('std', torch.tensor([]))
        self.fitted = False

    def fit(self, x: torch.Tensor):
        """
        Compute per‑feature mean and std from a [N, F]-shaped tensor.
        """
        # flatten any extra dims into the batch
        N = x.shape[0]
        feats = x.view(N, -1) if x.dim() > 2 else x
        self.mean = feats.mean(dim=0)
        self.std  = feats.std(dim=0, unbiased=False).clamp(min=1e-6)
        self.fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet")
        # preserve any extra dimensions beyond the feature-dim
        original_shape = x.shape
        N = x.shape[0]
        feats = x.view(N, -1) if x.dim() > 2 else x
        scaled = (feats - self.mean) / self.std
        return scaled.view(original_shape)

    # alias
    transform = forward



class DNN(nn.Module):
    def __init__(self, input_dim, output_dim=1, classification=False):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.classification = classification

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        if self.classification:
            x = torch.sigmoid(x)
        return x