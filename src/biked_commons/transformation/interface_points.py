import math

import pandas as pd
import torch


# Convert x to a PyTorch tensor


def calculate_interace_points_df(df):
    columns = ["Stack",
               "Handlebar style OHCLASS: 0", "Handlebar style OHCLASS: 1", "Handlebar style OHCLASS: 2",
               "Seat angle", "Saddle height", "Head tube length textfield", "Head tube lower extension2", 
               "Head angle", "DT Length"]
    x = df[columns].values
    x = torch.tensor(x, dtype=torch.float32)
    y = calculate_interface_points(x)
    y = pd.DataFrame(y.numpy(), columns=["hand_x", "hand_y", "hip_x", "hip_y", "crank_length"], index=df.index)
    return y


def calculate_interface_points(x, dtype=torch.float32):
    # TODO: consider one-hot compatible version?
    stack = x[:, 0]
    oh0 = x[:, 1]  # Handlebar style OHCLASS: 0
    oh1 = x[:, 2]  # Handlebar style OHCLASS: 1
    oh2 = x[:, 3]  # Handlebar style OHCLASS: 2

    mask0 = torch.logical_and(oh0 > oh1, oh0 > oh2).float()
    mask1 = torch.logical_and(oh1 > oh0, oh1 > oh2).float()
    mask2 = torch.logical_and(oh2 > oh0, oh2 > oh1).float()

    seat_angle = x[:, 4] * math.pi / 180  # Convert angles to radians
    saddle_height = x[:, 5]
    crank_length = torch.tensor(172.5) # TODO : make this a parameter once it is added to representation
    crank_length = crank_length * torch.ones_like(saddle_height)  # Crank length in mm

    HTL = x[:, 6]  # Head tube length textfield
    HTLX = x[:, 7]  # Head tube lower extension2
    HTA = x[:, 8] * math.pi / 180 # Head tube angle
    DTL = x[:, 9]  # DT Length
    DTJY = stack - (HTL - HTLX) * torch.sin(HTA)
    DTJX = torch.sqrt(DTL ** 2 - DTJY ** 2)
    handlebar_mount_x = DTJX - (HTL - HTLX) * torch.cos(HTA)
    handlebar_mount_y = stack

    handle_angle = torch.tensor(12.0 * math.pi / 180, dtype = dtype)
    # road_bar_reach = 80.0
    road_bar_drop = torch.tensor(128.0, dtype = dtype)  # mm
    hbarextend = torch.tensor(60.0, dtype = dtype)
    mountain_bar_sweep = torch.tensor(16.4, dtype = dtype)
    mountain_bar_rise = torch.tensor(10.0, dtype = dtype)
    mtndrop = torch.tensor(10.0, dtype = dtype)
    bullhorn_reach = torch.tensor(150, dtype = dtype)
    bullhorn_rise = torch.tensor(50.0, dtype = dtype)
    bullhorn_slant = torch.tensor(40*math.pi/180, dtype = dtype)  # radians


    #seat angle is angle between horizontal and line to saddle
    hip_x = saddle_height / torch.tan(seat_angle) 
    hip_y = saddle_height + 0.05  # hip_y (offset for hip not being directly on saddle)

    numfeat = x.shape[0]


    # posx and pos y are offset of hand position from handlbar mount
    posx = torch.zeros(numfeat, dtype=torch.float32)
    posy = torch.zeros(numfeat, dtype=torch.float32)

    # Precompute common trig terms
    cos_angle = torch.cos(handle_angle)
    sin_angle = torch.sin(handle_angle)

    #bullhorns
    mC = bullhorn_rise - 60 * torch.sin(bullhorn_slant + handle_angle)
    mA = bullhorn_reach - 60 * torch.cos(bullhorn_slant + handle_angle)
    posx += mask2 * (mA * cos_angle - mC * sin_angle)
    posy += mask2 * (mA * sin_angle + mC * cos_angle)

    #mountain bar
    v1 = mountain_bar_rise + mtndrop
    h1 = mountain_bar_sweep
    posx += mask1 * (-h1 * cos_angle - v1 * sin_angle)
    posy += mask1 * (-h1 * sin_angle + v1 * cos_angle)

    #drop bar
    v0 = road_bar_drop
    h0 = hbarextend - 60 
    posx += mask0 * (-h0 * cos_angle + v0 * sin_angle)
    posy += mask0 * (-h0 * sin_angle - v0 * cos_angle)


    hand_x = handlebar_mount_x + posx  # hand_x
    hand_y = handlebar_mount_y + posy  # hand_y
    

    y = [hand_x, hand_y, hip_x, hip_y, crank_length]
    y = torch.stack(y, dim=1)  # Stack the tensors along the second dimension
    y = y / 1000  # Convert to meters
    return y
