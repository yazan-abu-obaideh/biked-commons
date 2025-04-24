import numpy as np
import pandas as pd
import torch


from biked_commons.resource_utils import split_datasets_path

def sample_riders(num_samples, split = "test"):
    # Sample random riders from the rider data
    if split == "test":
        rider_data = pd.read_csv(split_datasets_path("aero_X_test.csv"), index_col=0)
        rider_data = rider_data[['upper_leg', 'lower_leg', 'arm_length', 'torso_length', 'neck_and_head_length', 'torso_width']]
    elif split == "train":
        rider_data = pd.read_csv(split_datasets_path("aero_X_train.csv"), index_col=0)
        rider_data = rider_data[['upper_leg', 'lower_leg', 'arm_length', 'torso_length', 'neck_and_head_length', 'torso_width']]
    else:
        raise ValueError("Invalid split. Choose 'train' or 'test'.")
    #sample num_samples with replacement
    sampled_riders = rider_data.sample(n=num_samples, replace=True).values
    return torch.tensor(sampled_riders, dtype=torch.float32)

def sample_use_case(num_samples, split=None):    
    # Randomly pick indices 0, 1 or 2
    idx = np.random.choice(3, size=num_samples, replace=True)
    
    # Convert to one-hot
    onehots = np.eye(3, dtype=int)[idx]
    
    return torch.tensor(onehots, dtype=torch.float32)

def sample_text(num_samples, split="test"):
    # read from .txt data into list of strings
    if split == "test":
        with open(split_datasets_path("text_descriptions_test.txt"), "r") as f:
            text_data = f.readlines()
    elif split == "train":
        with open(split_datasets_path("text_descriptions_train.txt"), "r") as f:
            text_data = f.readlines()
    else:
        raise ValueError("Invalid split. Choose 'train' or 'test'.")
    #select num_samples from list with replacement
    sampled_text = np.random.choice(text_data, size=num_samples, replace=True)
    
    return sampled_text.tolist()
