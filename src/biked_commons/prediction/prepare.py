import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_validity():
    df = pd.read_csv('../../resources/datasets/raw_datasets/validity.csv', index_col=0)
    df = df.reset_index(drop=True)
    subset = df[df['valid'].isin([0, 2])]
    Y = subset['valid']
    X = subset.drop('valid', axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

    X_train.to_csv('../../resources/datasets/split_datasets/validity_X_train.csv')
    X_test.to_csv('../../resources/datasets/split_datasets/validity_X_test.csv')
    Y_train.to_csv('../../resources/datasets/split_datasets/validity_Y_train.csv')
    Y_test.to_csv('../../resources/datasets/split_datasets/validity_Y_test.csv')

def prepare_structure():
    df = pd.read_csv('../../resources/datasets/raw_datasets/structure.csv', index_col=0)
    df = df.reset_index(drop=True)

    sim_1_displacements = df[["Sim 1 Dropout X Disp.", "Sim 1 Dropout Y Disp.", "Sim 1 Bottom Bracket X Disp.", "Sim 1 Bottom Bracket Y Disp."]].values
    sim_1_abs_displacements = np.abs(sim_1_displacements)
    sim_1_normalized_displacements = sim_1_abs_displacements / np.mean(sim_1_abs_displacements, axis=0)
    sim_1_compliance_score = np.mean(sim_1_normalized_displacements, axis=1)

    sim_2_displacements = df["Sim 2 Bottom Bracket Z Disp."].values
    sim_2_abs_displacements = np.abs(sim_2_displacements)
    sim_2_compliance_score = sim_2_abs_displacements / np.mean(sim_2_displacements)

    sim_3_displacements = df[["Sim 3 Bottom Bracket Y Disp.", "Sim 3 Bottom Bracket X Rot."]].values
    sim_3_abs_displacements = np.abs(sim_3_displacements)
    sim_3_normalized_displacements = sim_3_abs_displacements / np.mean(sim_3_abs_displacements, axis=0)
    sim_3_compliance_score = np.mean(sim_3_normalized_displacements, axis=1)

    mass = df["Model Mass"].values
    planar_SF = df["Sim 1 Safety Factor"].values
    eccentric_SF = df["Sim 3 Safety Factor"].values

    Y = np.stack([mass, sim_1_compliance_score, sim_2_compliance_score, sim_3_compliance_score, planar_SF, eccentric_SF], axis=1)
    Y = pd.DataFrame(Y, columns=["Mass", "Planar Compliance", "Transverse Compliance", "Eccentric Compliance", "Planar Safety Factor", "Eccentric Safety Factor"])
    X = df.drop(["Model Mass", "Sim 1 Dropout X Disp.", "Sim 1 Dropout Y Disp.", "Sim 1 Bottom Bracket X Disp.", "Sim 1 Bottom Bracket Y Disp.", "Sim 2 Bottom Bracket Z Disp.", "Sim 3 Bottom Bracket Y Disp.", "Sim 3 Bottom Bracket X Rot.", "Sim 1 Safety Factor", "Sim 3 Safety Factor"], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    X_train.to_csv('../../resources/datasets/split_datasets/structure_X_train.csv')
    X_test.to_csv('../../resources/datasets/split_datasets/structure_X_test.csv')
    Y_train.to_csv('../../resources/datasets/split_datasets/structure_Y_train.csv')
    Y_test.to_csv('../../resources/datasets/split_datasets/structure_Y_test.csv')

def prepare_aero():
    df = pd.read_csv('../../resources/datasets/raw_datasets/aero.csv', index_col=0)
    df = df.reset_index(drop=True)

    Y = df["Drag"]
    X = df.drop("Drag", axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    X_train.to_csv('../../resources/datasets/split_datasets/aero_X_train.csv')
    X_test.to_csv('../../resources/datasets/split_datasets/aero_X_test.csv')
    Y_train.to_csv('../../resources/datasets/split_datasets/aero_Y_train.csv')
    Y_test.to_csv('../../resources/datasets/split_datasets/aero_Y_test.csv')



