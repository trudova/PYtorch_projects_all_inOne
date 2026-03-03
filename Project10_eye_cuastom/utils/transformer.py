import torchvision.transforms as transforms


def transform():
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform
