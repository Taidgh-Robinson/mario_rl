import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.convLayer1 = nn.Conv2d(in_channels=18, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.convLayer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.convLayer3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.convLayer4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0)
        self.linearLayer1 = nn.Linear(53760, 32768)
        self.linearLayer2 = nn.Linear(32768, 8192)
        self.linearLayer3 = nn.Linear(8192, 512)
        self.value_stream = nn.Linear(512, 1)  # Produces V(s)
        self.advantage_stream = nn.Linear(512, n_actions)  # Produces A(s, a) for each action

    def forward(self, x):
        #Feed Through Convolution Layers
        x = F.relu(self.convLayer1(x))
        x = F.relu(self.convLayer2(x))
        x = F.relu(self.convLayer3(x))
        x = F.relu(self.convLayer4(x))
        #Flatten input
        flatened_x = x.view(-1)
        #Feed Through Linear layers
        x = F.relu(self.linearLayer1(flatened_x))
        x = F.relu(self.linearLayer2(x))
        x = F.relu(self.linearLayer3(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage to get Q-values
        q_values = value + advantage - advantage.mean(dim=0, keepdim=True)

        return q_values
