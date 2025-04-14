import dill
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

from biked_commons.resource_utils import resource_path
from biked_commons.usability.mlp_model import MLP
from biked_commons.usability.usability_ordered_columns import ORDERED_COLUMNS


_SVM_MODEL_PATH = resource_path("models/svm_model.pkl")
_MLP_MODEL_PATH = resource_path("models/mlp_with_hyperparameters.pth")
_DATA_PATH = resource_path("datasets/raw_datasets/clip_sBIKED_processed.csv")
_SCALER_PATH = resource_path("scaler_usability.pk")


def _load_scaler() -> preprocessing.MinMaxScaler:
    try:
        with open(_SCALER_PATH, "rb") as file:
            return dill.load(file)
    except FileNotFoundError:
        df = pd.read_csv(_DATA_PATH, index_col=0)[ORDERED_COLUMNS]
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(df.values)
        with open(_SCALER_PATH, "wb") as file:
            dill.dump(scaler, file)
        return scaler


def _load_svm_model():
    return joblib.load(_SVM_MODEL_PATH)


def _load_mlp_model():
    checkpoint = torch.load(_MLP_MODEL_PATH)
    hp = checkpoint["hyperparameters"]
    model = MLP(
        input_dim=hp["input_dim"],
        hidden_dims=hp["hidden_dims"],
        dropout_rate=hp["dropout_rate"],
        lr=hp["lr"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


class UsabilityPredictorBinary:
    """Predicts binary usability. Works on unscaled input."""

    def __init__(self):
        self._scaler = _load_scaler()           
        self._model = _load_svm_model()        

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x_unscaled = self._inverse_scale_if_needed(x)
        return self._predict(x_unscaled)

    def _inverse_scale_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if x.min() >= 0 and x.max() <= 1:
            x_np = x.detach().cpu().numpy()
            x_inv = self._scaler.inverse_transform(x_np)
            return torch.tensor(x_inv, dtype=torch.float32, device=x.device)
        return x

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        predictions = self._model.predict(x_np)
        return torch.tensor(predictions, dtype=torch.float32, device=x.device)


class UsabilityPredictorContinuous:
    """Predicts continuous usability. Works on scaled input."""
    def __init__(self):
        self._scaler = _load_scaler()
        self._model = _load_mlp_model()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict(self._scale_if_needed(x))

    def _scale_if_needed(self, x: torch.Tensor):
        if torch.any(x < 0) or torch.any(x > 1):
            x_np = x.detach().cpu().numpy()  
            x_scaled_np = self._scaler.transform(x_np)
            return torch.from_numpy(x_scaled_np).to(device= x.device, dtype=torch.float32)
        return x

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)