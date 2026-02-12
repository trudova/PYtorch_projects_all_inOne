import torch
import torchvision
from ImmageClassificationNet import ImmageClassificationNet
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from utils.device import device_detection, gpu_load
from utils.transformer import transform

gpu_load()
device = device_detection()
transform = transform()

batch_size = 4

testset = torchvision.datasets.ImageFolder(root="./data/test", transform=transform)
test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

model = ImmageClassificationNet()
model_state_dict = torch.load("model_4.pth", weights_only=True, map_location=device)
model.load_state_dict(model_state_dict)
model = model.to(device)

y_test = []
y_pred = []

model.eval()
with torch.no_grad():
    for b, (X_test, y_test_temp) in enumerate(test_loader):
        X_test = X_test.to(device)
        y_test_temp = y_test_temp.to(device)
        y_val = model(X_test).round()
        y_test.extend(y_test_temp.cpu().numpy())
        y_pred.extend(y_val.cpu().numpy())

acc = accuracy_score(y_test, y_pred)
print("FINAL accurasy", acc * 100, "%")
