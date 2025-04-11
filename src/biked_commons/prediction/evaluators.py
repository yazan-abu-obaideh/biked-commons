import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, r2_score, mean_squared_error

def evaluate_frame_validity(model):
    X_test = pd.read_csv('../../resources/datasets/split_datasets/validity_X_test.csv', index_col=0)    
    Y_test = pd.read_csv('../../resources/datasets/split_datasets/validity_Y_test.csv', index_col=0)
    predictions = model(X_test)
    return f1_score(Y_test, predictions)

def evaluate_structure(model):
    Y_test = pd.read_csv('../../resources/datasets/split_datasets/structure_Y_test.csv', index_col=0)
    X_test = pd.read_csv('../../resources/datasets/split_datasets/structure_X_test.csv', index_col=0)
    predictions = model(X_test)
    return r2_score(Y_test, predictions)

def evaluate_aero(model):
    Y_test = pd.read_csv('../../resources/datasets/split_datasets/aero_Y_test.csv', index_col=0)
    X_test = pd.read_csv('../../resources/datasets/split_datasets/aero_X_test.csv', index_col=0)
    predictions = model(X_test)
    return r2_score(Y_test, predictions)

def evaluate_clip(model):
    Y_test = pd.read_csv("../../resources/datasets/split_datasets/CLIP_Y_test.npy", index_col=0)
    X_test = np.load("../../resources/datasets/split_datasets/CLIP_X_test.npy")
    predictions = model(X_test)
    return mean_squared_error(Y_test, predictions)