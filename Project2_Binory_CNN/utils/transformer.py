import torchvision.transforms as transforms


def transform():
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform
