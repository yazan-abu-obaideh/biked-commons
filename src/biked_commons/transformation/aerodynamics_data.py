import torch
import pandas as pd
import numpy as np
import math

def assemble_aero_data(interface_df, anthropometric_df):

    #concat in dim 1
    combined_df = pd.concat([interface_df, anthropometric_df], axis=1)
    combined_df.columns = ['hand_x', 'hand_y', 'hip_x', 'hip_y', 'crank_length', 'upper_leg', 'lower_leg', 'arm_length', 'torso_length', 'neck_and_head_length', 'torso_width']

def leg_long_enough(x):
    ul, ll = x['upper_leg'], x['lower_leg']
    Lspl = torch.sqrt(x['hip_x'] ** 2 + x['hip_y'] ** 2) + x['crank_length']
    return ul + ll >= Lspl

def saddle_far_enough_from_handles(x):
    arm, back = x['arm_length'], x['torso_length']
    Lsh = torch.sqrt((x['hip_x'] + x['hand_x']) ** 2 + (x['hip_y'] - x['hand_y']) ** 2)
    return arm + back >= Lsh

def saddle_far_enough_from_bottom_bracket(x):
    return torch.abs(x['lower_leg'] - x['upper_leg']) <= torch.sqrt(x['hip_x'] ** 2 + x['hip_y'] ** 2) - x['crank_length']

def hip_angle_over_n_degrees(x, n=10):
    rad2deg = 180 / math.pi
    Lsh = torch.sqrt((x['hip_x'] + x['hand_x']) ** 2 + (x['hip_y'] - x['hand_y']) ** 2)
    
    # Angle from saddle to handlebar in degrees
    Tsh = torch.atan((x['hand_y'] - x['hip_y']) / (x['hip_x'] + x['hand_x'])) * rad2deg
    
    # Angle from saddle to pedals in degrees
    Tsp = torch.atan(-x['hip_y'] / x['hip_x']) * rad2deg
    
    # Angle from shoulders to saddle to handlebar
    Tssh = torch.acos((x['torso_length'] ** 2 + Lsh ** 2 - x['arm_length'] ** 2) / (2 * x['torso_length'] * Lsh)) * rad2deg
    
    # Angle from knee to saddle to closest pedal position
    Tksp = torch.acos((x[:, 1] ** 2 + (torch.sqrt(x['hip_x'] ** 2 + x['hip_y'] ** 2) - x['crank_length']) ** 2 - x['lower_leg'] ** 2) / (2 * x['upper_leg'] * (torch.sqrt(x['hip_x'] ** 2 + x['hip_y'] ** 2) - x['crank_length']))) * rad2deg
    
    return (Tsh + Tssh) - (Tsp + Tksp) >= n

def knee_angle_over_n_degrees(x, n=20):
    rad2deg = 180 / math.pi = 180 / math.pi
    ul, ll = x['upper_leg'], x['lower_leg']
    Lsps = torch.sqrt(x['hip_x'] ** 2 + x['hip_y'] ** 2) - x['crank_length']
    
    # Angle from saddle to knee to closest pedal position
    Tskp = torch.acos((ul ** 2 + ll ** 2 - Lsps ** 2) / (2 * ul * ll)) * rad2deg = 180 / math.pi
    
    return Tskp >= n

def checkconfig(x):
    valid = leg_long_enough(x) & saddle_far_enough_from_handles(x) & saddle_far_enough_from_bottom_bracket(x) & hip_angle_over_n_degrees(x) & knee_angle_over_n_degrees(x)
    valid &= torch.all(x > 0, dim=1)
    return valid
