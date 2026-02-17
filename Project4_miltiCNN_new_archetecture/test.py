import torch
import torch.nn as nn
import torchvision
from Data import get_test_loader
from Model import get_model
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.Device import device_detection, gpu_load
from utils.transformer import transform

device = device_detection()
gpu_load()

test_loader = get_test_loader()

y_test = []
y_pred = []

resnet50_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)
resnet50_model.fc = torch.nn.Identity()  # remove the final classification layer

resnet50_model = resnet50_model.to(device)


fc_model = get_model()
fc_state_dict = torch.load("fc_model_2.pth", weights_only=True, map_location=device)
fc_model.load_state_dict(fc_state_dict)
fc_model = fc_model.to(device)

model = torch.nn.Sequential(resnet50_model, fc_model)
model = model.to(device)


model.eval()
with torch.no_grad():
    for b, (X_test, y_test_temp) in enumerate(test_loader):
        X_test = X_test.to(device)
        y_test_temp = y_test_temp.to(device)
        y_val = nn.functional.softmax(model(X_test), dim=1)  # logits shape [batch, 3]
        predicted = torch.argmax(y_val, dim=1)  # class ids shape [batch]

        y_test.extend(y_test_temp.cpu().numpy().reshape(-1))
        y_pred.extend(predicted.cpu().numpy().reshape(-1))

acc = accuracy_score(y_test, y_pred)
print("FINAL accuracy", acc * 100, "%")

cmx = confusion_matrix(y_test, y_pred)
CLASSES = ["affenpinscher", "akita", "corgi"]
print(CLASSES)
print(cmx)

print("FINAL accuracy", acc * 100, "%")

model.eval()
preprocess = transform()
corgi = Image.open("image.jpg").convert("RGB")
corgi = preprocess(corgi).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(corgi)
    pred_id = torch.argmax(output, dim=1).item()
    print("Predicted class:", CLASSES[pred_id])
