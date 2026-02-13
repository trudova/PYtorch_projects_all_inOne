import torch
import torchvision
from ImageMultiClassCNN import ImageMultiClassCNN
from torch.utils.data import DataLoader
from utils.device import device_detection, gpu_load
from utils.transformer import transform

gpu_load()
device = device_detection()
transform = transform()

model = ImageMultiClassCNN()
model_state_dict = torch.load("model_7.pth", weights_only=True, map_location=device)
model.load_state_dict(model_state_dict)
model = model.to(device)
batch_size = 4

testset = torchvision.datasets.ImageFolder(root="./data/test", transform=transform)
test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)


CLASSES = ["affenpinscher", "akita", "corgi"]
y_test = []
y_pred = []

model.eval()
with torch.no_grad():
    for b, (X_test, y_test_temp) in enumerate(test_loader):
        X_test = X_test.to(device)
        y_test_temp = y_test_temp.to(device)
        y_val = model(X_test)  # logits shape [batch, 3]
        predicted = torch.argmax(y_val, dim=1)  # class ids shape [batch]

        y_test.extend(y_test_temp.cpu().numpy().reshape(-1))
        y_pred.extend(predicted.cpu().numpy().reshape(-1))
