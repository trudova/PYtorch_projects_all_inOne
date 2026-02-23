import torch.nn as nn
import torchvision
from utils.Device import device_detection

device = device_detection()


def get_fc_model():
    fc_model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.1),
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
    return fc_model.to(device)


def get_densNet_model():
    densNet_model = torchvision.models.densenet121(
        weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
    )
    for params in densNet_model.parameters():
        params.requires_grad = False

    densNet_model.classifier = get_fc_model()  # remove the final classification layer

    densNet_model = densNet_model.to(device)
    return densNet_model
