import torch
from torch import nn
import numpy as np

class MLPLengthRegressor(nn.Module):
    def __init__(self, text_dim, mean_std_file, dropout=0.2, hidden_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(text_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)
        self.mean, self.std = np.load(mean_std_file)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def denormalize(self, x):
        assert x.shape[1] == 2  # mean and logvar
        old_std = torch.sqrt(torch.exp(x[:, 1]))
        old_mean = x[:, 0]
        new_std = self.std * old_std
        new_mean = self.std * old_mean + self.mean
        out = torch.stack([new_mean, new_std], dim=1)
        return out