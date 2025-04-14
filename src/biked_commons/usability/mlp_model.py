import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate, lr):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  
        self.network = nn.Sequential(*layers)
        
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.network(x)

