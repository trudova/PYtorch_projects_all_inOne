import torch.nn as nn
import torch.nn.functional as F


class LoanClassificationNN(nn.Module):
    def __init__(self, num_features, hidden_features1=20, hidden_features2=10):
        super().__init__()
        self.layer1 = nn.Linear(num_features, hidden_features1)
        self.layer2 = nn.Linear(hidden_features1, hidden_features2)
        self.layer3 = nn.Linear(hidden_features2, 5)
        self.layer4 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x
