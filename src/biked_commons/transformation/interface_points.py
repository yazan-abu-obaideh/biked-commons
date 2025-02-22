import math

import pandas as pd
import torch


# Convert x to a PyTorch tensor


def calculate_interace_points_df(df):
    columns = ["Components/General_Handlebar_from_BB_(X)", "Components/General_Handlebar_from_BB_(Y)",
               "Components/Handlebar_Style", "Components/Handlebar_A", "Components/Handlebar_B",
               "Components/Handlebar_C", "Components/Handlebar_angle",
               "Components/General_Saddle_from_BB_(X)", "Components/General_Saddle_from_BB_(Y)", "Components/Cranks_A"]
    x = df[columns].values
    x = torch.tensor(x, dtype=torch.float32)
    y = calculate_interface_points(x)
    y = pd.DataFrame(y.numpy(), columns=["hand_x", "hand_y", "hip_x", "hip_y", "crank_length"], index=df.index)
    return y


def calculate_interface_points(x, device="cpu", dtype=torch.float32):
    # TODO: consider one-hot compatible version?

    numfeat = x.shape[0]

    # Extract relevant columns
    ht = x[:, 2]
    ang = x[:, 6] * math.pi / 180  # Convert angles to radians

    # Initialize posx and posy tensors
    posx = torch.zeros(numfeat, dtype=torch.float32)
    posy = torch.zeros(numfeat, dtype=torch.float32)

    # Case ht == 2
    mask2 = ht == 2
    mC = x[:, 5] / 2
    mA = x[:, 3] - 40
    posx[mask2] = mA[mask2] * torch.cos(ang[mask2]) - mC[mask2] * torch.sin(ang[mask2])
    posy[mask2] = mA[mask2] * torch.sin(ang[mask2]) + mC[mask2] * torch.cos(ang[mask2])

    # Case ht == 1
    mask1 = ht == 1
    v1 = x[:, 4] + x[:, 5]
    h1 = x[:, 3]
    posx[mask1] = -h1[mask1] * torch.cos(ang[mask1]) - v1[mask1] * torch.sin(ang[mask1])
    posy[mask1] = -h1[mask1] * torch.sin(ang[mask1]) + v1[mask1] * torch.cos(ang[mask1])

    # Case ht == 0
    mask0 = ht == 0
    v0 = x[:, 4]
    h0 = x[:, 5] - 60
    posx[mask0] = -h0[mask0] * torch.cos(ang[mask0]) + v0[mask0] * torch.sin(ang[mask0])
    posy[mask0] = -h0[mask0] * torch.sin(ang[mask0]) - v0[mask0] * torch.cos(ang[mask0])

    # Construct y tensor
    y = torch.zeros((numfeat, 5), dtype=dtype, device=device)
    y[:, 0] = x[:, 0] + posx  # hand_x
    y[:, 1] = x[:, 1] + posy  # hand_y
    y[:, 2] = x[:, 7]  # hip_x
    y[:, 3] = x[:, 8] + 0.05  # hip_y (offset for hip not being directly on saddle)
    y[:, 4] = x[:, 9]  # crank_length

    y = y / 1000  # Convert to meters
    return y
