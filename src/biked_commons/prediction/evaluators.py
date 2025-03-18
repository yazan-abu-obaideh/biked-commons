import pandas as pd
from sklearn.metrics import f1_score, r2_score

def evaluate_frame_validity(predictions):
    Y_test = pd.read_csv('../../resources/datasets/split_datasets/Y_test.csv', index_col=0)
    return f1_score(Y_test, predictions)

def evaluate_structure(predictions):
    Y_test = pd.read_csv('../../resources/datasets/split_datasets/Y_test.csv', index_col=0)
    return r2_score(Y_test, predictions)

def evaluate_aero(predictions):
    Y_test = pd.read_csv('../../resources/datasets/split_datasets/Y_test.csv', index_col=0)
    return r2_score(Y_test, predictions)