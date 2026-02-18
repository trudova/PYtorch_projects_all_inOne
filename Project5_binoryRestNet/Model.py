import torch
import torch.nn as nn
import torchvision
from utils.Device import device_detection

device = device_detection()


def get_model():
    model = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.20),
        nn.Linear(64, 12),
        nn.BatchNorm1d(12),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(12, 1),
    )
    return model.to(device)


def get_resNet50_model():
    resnet50_model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    )
    resnet50_model.fc = torch.nn.Identity()  # remove the final classification layer

    resnet50_model = resnet50_model.to(device)
    return resnet50_model
