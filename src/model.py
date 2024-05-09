import torch.nn as nn
import torch.nn.functional as F

class QR_DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_quantiles):
        super(QR_DQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions * num_quantiles)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        quantiles = self.fc3(x)
        return quantiles
