import torchvision.transforms as transforms


def transform():
    transform = transforms.Compose(
        [
            # transforms.Resize((256, 256)),
            # transforms.RandomInvert(),
            # transforms.RandomRotation(10),
            # transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(128),
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform
