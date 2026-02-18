import numpy as np
import torch
import torchvision
from Data import get_test_loader
from Model import get_model
from PIL import Image
from utils.Device import device_detection, gpu_load
from utils.transformer import transform

device = device_detection()
gpu_load()

test_loader = get_test_loader()

CLASSES = ["defective", "good"]

resnet50_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)
resnet50_model.fc = torch.nn.Identity()  # remove the final classification layer

resnet50_model = resnet50_model.to(device)


fc_model = get_model()
fc_state_dict = torch.load("fc_model_5.pth", weights_only=True, map_location=device)
fc_model.load_state_dict(fc_state_dict)
fc_model = fc_model.to(device)

model = torch.nn.Sequential(resnet50_model, fc_model)
model = model.to(device)

model.eval()
preprocess = transform()
tire = Image.open("./test/tire3.jpg").convert("RGB")
tire = preprocess(tire).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(tire)
    pred_id = torch.sigmoid(output).item()
    print("Predicted class:", CLASSES[int(np.round(pred_id))])
