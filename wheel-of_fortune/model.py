import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size=4, action_size=14, seed=1111):
        """
        Initialize Deep Q Network

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        #self.conv1 = nn.Conv2d(state_size, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        #self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        #self.fc4 = nn.Linear(7 * 7 * 64, 512)
        #self.head = nn.Linear(512, action_size)

        fc1_size = 128
        fc2_size = 128
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.out = nn.Linear(fc2_size, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        #state = state.float() / 255
        #state = F.relu(self.conv1(state))
        #state = F.relu(self.conv2(state))
        #state = F.relu(self.conv3(state))
        #state = F.relu(self.fc4(state.view(state.size(0), -1)))
        #return self.head(state)

        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action = self.out(x)
        return action
