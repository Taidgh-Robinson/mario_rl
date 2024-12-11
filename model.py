import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.convLayer1 = nn.Conv2d(in_channels=18, out_channels=64, kernel_size=8, stride=4, padding=0)
    
    def forward(self, x):
        x = F.relu(self.convLayer1(x))
        return x 
