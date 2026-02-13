import torchvision
from torch.utils.data import DataLoader
from utils.Device import device_detection
from utils.transformer import transform

device = device_detection()
transform = transform()

batch_size = 2
trainset = torchvision.datasets.ImageFolder(root="./data/train", transform=transform)
testset = torchvision.datasets.ImageFolder(root="./data/test", transform=transform)


def get_trainset():
    return trainset


def get_testset():
    return testset


def get_train_loader():
    train_loader = DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True, pin_memory_device=device
    )
    return train_loader


def get_test_loader():
    test_loader = DataLoader(
        dataset=testset, batch_size=batch_size, shuffle=True, pin_memory_device=device
    )
    return test_loader
