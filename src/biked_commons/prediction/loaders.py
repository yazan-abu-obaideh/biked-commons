import pandas as pd

def load_frame_validity():
    X_train = pd.read_csv('../../resources/datasets/split_datasets/validity_X_train.csv', index_col=0)
    X_test = pd.read_csv('../../resources/datasets/split_datasets/validity_X_test.csv', index_col=0)
    Y_train = pd.read_csv('../../resources/datasets/split_datasets/validity_Y_train.csv', index_col=0)
    return X_train, X_test, Y_train

def load_structure():
    X_train = pd.read_csv('../../resources/datasets/split_datasets/structure_X_train.csv', index_col=0)
    X_test = pd.read_csv('../../resources/datasets/split_datasets/structure_X_test.csv', index_col=0)
    Y_train = pd.read_csv('../../resources/datasets/split_datasets/structure_Y_train.csv', index_col=0)
    return X_train, X_test, Y_train

def load_aero():
    X_train = pd.read_csv('../../resources/datasets/split_datasets/aero_X_train.csv', index_col=0)
    X_test = pd.read_csv('../../resources/datasets/split_datasets/aero_X_test.csv', index_col=0)
    Y_train = pd.read_csv('../../resources/datasets/split_datasets/aero_Y_train.csv', index_col=0)
    return X_train, X_test, Y_train