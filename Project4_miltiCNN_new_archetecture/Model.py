import torch.nn as nn
from utils.Device import device_detection
from utils.transformer import transform

device = device_detection()
transform = transform()


def get_model():
    model = nn.Sequential(
        nn.Sequential(
            nn.Conv2d(3, 6, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Dropout(0.1),
        ),
        nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
        ),
        nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        ),
        nn.Flatten(),
        nn.Sequential(
            nn.Linear(32 * 8 * 8, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 3),
        ),
    )
    return model.to(device)
