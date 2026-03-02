import torchvision.transforms as transforms


def get_transform():
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomInvert(),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(
            #     brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            # ),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform
