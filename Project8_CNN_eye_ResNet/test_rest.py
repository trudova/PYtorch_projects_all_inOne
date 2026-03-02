import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
from fc_resnet_model import get_model
from get_data import get_test_loader, get_testset, get_trainset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from utils.Device import device_detection, gpu_load
from utils.transformer import transform

CLASSES = [
    "No_ulcer_of_the_corneal_epithelium",
    "Micro_punctate",
    "Macro_punctate",
    "Coalescent_macro_punctate",
    # "Patch_ge_1mm",
]
NUM_CLASSES = len(CLASSES)

gpu_load()
device = device_detection()
transform = transform()


trainset = get_trainset()
testset = get_testset()

test_loader = get_test_loader()


resnet152_model = torchvision.models.resnet152(
    weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1
)

resnet152_model.fc = torch.nn.Identity()  # remove the final classification layer

resnet152_model = resnet152_model.to(device)

fc_model = get_model()
fc_state_dict = torch.load("model.pth", weights_only=True, map_location=device)
fc_model.load_state_dict(fc_state_dict)
fc_model = fc_model.to(device)


model = torch.nn.Sequential(resnet152_model, fc_model)
model = model.to(device)

y_test = []
y_pred = []

model.eval()
with torch.no_grad():
    for b, (X_test, y_test_temp) in enumerate(get_test_loader()):
        X_test = X_test.to(device)
        y_test_temp = y_test_temp.to(device)
        y_val = model(X_test)
        predicted = torch.argmax(y_val, dim=1)

        y_test.extend(y_test_temp.cpu().numpy().reshape(-1))
        y_pred.extend(predicted.cpu().numpy().reshape(-1))


acc = accuracy_score(y_test, y_pred)
print(f"accuracy is {acc * 100}%")
f1 = f1_score(y_test, y_pred, average="macro")  # important for multiclass
print(f"F1 score: {f1:.4f}")
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, xticklabels=CLASSES, yticklabels=CLASSES)
