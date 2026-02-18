import torchvision
from torch.utils.data import DataLoader, random_split
from utils.Device import device_detection
from utils.transformer import transform

device = device_detection()
transform = transform()

batch_size = 16
dataset = torchvision.datasets.ImageFolder(root="./images", transform=transform)

trainset, testset = random_split(dataset, [0.8, 0.2])


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
