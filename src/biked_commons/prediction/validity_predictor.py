import torch
import torch.nn as nn
import math


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
        x = (x - self.mean) / self.std #normalize features
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x