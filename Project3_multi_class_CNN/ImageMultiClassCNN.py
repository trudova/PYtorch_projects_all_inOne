import torch.nn as nn
import torch.nn.functional as F


class ImageMultiClassCNN(nn.Module):
    def __init__(self):
        super(ImageMultiClassCNN, self).__init__()
        #  color chanal, 6 filters(out channels), 3x3 kernel, stride(step) 1, padding 1
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.conv2 = nn.Conv2d(6, 16, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 1, 1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax(x)
        return x
