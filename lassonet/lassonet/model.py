import torch 
import torch.nn as nn 
import torch.nn.functional as F

class LassoNet(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(input))
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, output)
        self.M = 10.0

    def forward(self, x):
        lin = torch.matmul(x, self.theta)
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return lin.unsqueeze(1) + out
    

