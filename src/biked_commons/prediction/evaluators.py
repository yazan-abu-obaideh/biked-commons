import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, r2_score, mean_squared_error

def evaluate_frame_validity(model, device="cpu"):
    X_test = pd.read_csv('../../resources/datasets/split_datasets/validity_X_test.csv', index_col=0)    
    Y_test = pd.read_csv('../../resources/datasets/split_datasets/validity_Y_test.csv', index_col=0)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    predictions = model(X_test_tensor).detach().cpu().numpy()
    return f1_score(Y_test, predictions)

def evaluate_structure(model, device="cpu"):
    Y_test = pd.read_csv('../../resources/datasets/split_datasets/structure_Y_test.csv', index_col=0)
    X_test = pd.read_csv('../../resources/datasets/split_datasets/structure_X_test.csv', index_col=0)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    predictions = model(X_test_tensor).detach().cpu().numpy()
    return r2_score(Y_test, predictions)

def evaluate_aero(model, device="cpu"):
    Y_test = pd.read_csv('../../resources/datasets/split_datasets/aero_Y_test.csv', index_col=0)
    X_test = pd.read_csv('../../resources/datasets/split_datasets/aero_X_test.csv', index_col=0)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    predictions = model(X_test_tensor).detach().cpu().numpy()
    return r2_score(Y_test, predictions)

def evaluate_clip(model, device="cpu"):
    Y_test = pd.read_csv("../../resources/datasets/split_datasets/CLIP_Y_test.npy", index_col=0)
    X_test = np.load("../../resources/datasets/split_datasets/CLIP_X_test.npy")
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    predictions = model(X_test_tensor).detach().cpu().numpy()
    return mean_squared_error(Y_test, predictions)