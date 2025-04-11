import torch
import torch.nn as nn
import math

def calculate_features(X):
    def law_of_cos(a, b, c):
        return (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)

    hand_x = X[:, 0]
    hand_y = X[:, 1]
    hip_x = X[:, 2]
    hip_y = X[:, 3]
    crank_length = X[:, 4]
    upper_leg = X[:, 5]
    lower_leg = X[:, 6]
    arm_length = X[:, 7]
    torso_length = X[:, 8]
    neck_and_head_length = X[:, 9]
    torso_width = X[:, 10]

    #some parameters as defined in the CAD model
    head_diameter = 0.25
    lower_leg_width = 0.12
    arm_width = 0.1
    upper_leg_width = (torso_width/2 - 0.16)/2 + 0.14
    neck_width = 0.12

    head_surface_area = head_diameter * head_diameter * math.pi/4 *torch.ones_like(neck_and_head_length) # surface area of the head

    hand_hip_horizontal_angle = torch.atan2(hip_y - hand_y, hip_x + hand_x) # angle from the hand to the hip to horizontal

    hand_hip_distance = torch.sqrt((hand_x + hip_x) ** 2 + (hand_y - hip_y) ** 2) # distance from the hand to the hip

    shoulder_hip_hand_angle = law_of_cos(hand_hip_distance, torso_length, arm_length) # angle from the shoulder to the hip to the hand

    shoulder_hip_horizontal_angle = hand_hip_horizontal_angle + shoulder_hip_hand_angle # angle from the shoulder to the hip to horizontal
    
    height_hip_shoulders = shoulder_hip_horizontal_angle * torso_length # vertical height from the shoulders to the handlebar

    torso_surface_area = torso_width * height_hip_shoulders # surface area of the torso

    height_handle_shoulders = height_hip_shoulders - hand_y + hip_y # vertical height from the handlebar to the shoulders

    arm_surface_area = arm_width * height_handle_shoulders # surface area of the arm

    neck_surface_area = neck_width * (neck_and_head_length - head_diameter) # surface area of the neck

    # height from the hip to the foot
    high_leg_hip_foot_dist = torch.sqrt((hip_x) ** 2 + (hip_y - crank_length) ** 2) # leg on the higher pedal
    low_leg_hip_foot_dist = torch.sqrt((hip_x) ** 2 + (hip_y + crank_length) ** 2) # leg on the lower pedal

    #angle from knee to hip to foot
    high_leg_knee_hip_foot_angle = law_of_cos(high_leg_hip_foot_dist, upper_leg, lower_leg) # leg on the higher pedal
    low_leg_knee_hip_foot_angle = law_of_cos(low_leg_hip_foot_dist, upper_leg, lower_leg) # leg on the lower pedal

    #angle from foot to hip to horizontal
    high_leg_foot_hip_horizontal_angle = torch.atan2(hip_y - crank_length, hip_x) # leg on the higher pedal
    low_leg_foot_hip_horizontal_angle = torch.atan2(hip_y + crank_length, hip_x) # leg on the lower pedal

    #angle from knee to hip to horizontal
    high_leg_knee_hip_horizontal_angle = high_leg_foot_hip_horizontal_angle - high_leg_knee_hip_foot_angle # leg on the higher pedal
    low_leg_knee_hip_horizontal_angle = low_leg_foot_hip_horizontal_angle - low_leg_knee_hip_foot_angle # leg on the lower pedal

    # mask for whether knee is below hip. 
    high_leg_knee_below_hip_mask = (high_leg_knee_hip_horizontal_angle > 0).float()
    low_leg_knee_below_hip_mask = (low_leg_knee_hip_horizontal_angle > 0).float() 

    # vertical height from the knee to the hip
    high_leg_height_knee_hip = torch.sin(high_leg_knee_hip_horizontal_angle)  #leg on the higher pedal
    low_leg_height_knee_hip = torch.sin(low_leg_knee_hip_horizontal_angle) #leg on the lower pedal

    # vertical height from the knee to the foot 
    high_leg_height_knee_foot = hip_y - crank_length - high_leg_height_knee_hip # leg on the higher pedal
    low_leg_height_knee_foot = hip_y + crank_length - low_leg_height_knee_hip # leg on the lower pedal

    # surface area of the lower portion of the leg 
    high_leg_lower_leg_surface_area = high_leg_height_knee_foot * lower_leg_width # leg on the higher pedal
    low_leg_lower_leg_surface_area = low_leg_height_knee_foot * lower_leg_width # leg on the lower pedal

    # surface area of the upper portion of the leg on the higher pedal. If knee is above hip, we need to use the upper leg width minus the lower leg width to calculate the added area of the upper leg. 
    high_leg_upper_leg_surface_area = high_leg_height_knee_hip * (upper_leg_width - high_leg_knee_below_hip_mask * lower_leg_width) # leg on the higher pedal
    low_leg_upper_leg_surface_area = low_leg_height_knee_hip * (upper_leg_width - low_leg_knee_below_hip_mask * lower_leg_width) # leg on the lower pedal

    # surface area of the leg
    high_leg_surface_area = high_leg_upper_leg_surface_area + high_leg_lower_leg_surface_area # leg on the higher pedal
    low_leg_surface_area = low_leg_upper_leg_surface_area + low_leg_lower_leg_surface_area #leg on the lower pedal

    leg_surface_area = high_leg_surface_area + low_leg_surface_area # total surface area of the legs

    frontal_surface_area = torso_surface_area + leg_surface_area + head_surface_area + arm_surface_area + neck_surface_area

    features = torch.stack([frontal_surface_area, torso_surface_area, leg_surface_area, arm_surface_area, neck_surface_area])

    return features.T


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, mean, std):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.mean = mean
        self.std = std

    def forward(self, x):
        # Normalize the input
        features = calculate_features(x)
        x = torch.cat((x, features), dim=1)
        x = (x - self.mean) / self.std
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x