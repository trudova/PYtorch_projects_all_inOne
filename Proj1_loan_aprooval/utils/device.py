import torch


def device_detection():
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def gpu_load():
    weights = torch.load("model_3.pth", weights_only=True, map_location="cpu")
    torch.save(weights, "model_3.pth")
