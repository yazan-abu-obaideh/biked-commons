import torch
import torch.nn as nn
import math
import pandas as pd

from biked_commons.resource_utils import models_and_scalers_path
from biked_commons.prediction.prediction_utils import TorchStandardScaler

class ValidityPreprocessor(nn.Module):
    def __init__(self, device: torch.device = None):
        super().__init__()
        scaler_path = models_and_scalers_path("validity_scaler.pt")
        self.device = device or torch.device('cpu')
        self.scaler: TorchStandardScaler = torch.load(scaler_path, map_location=self.device)
        self.scaler.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler(x)

    __call__ = forward
