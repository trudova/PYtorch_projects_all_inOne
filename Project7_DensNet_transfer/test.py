import seaborn as sns
import torch
from Data import get_test_loader
from matplotlib import pyplot as plt
from Model import get_densNet_model
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from utils.Device import device_detection, gpu_load

CLASSES = ["cat", "dog"]
NUM_CLASSES = len(CLASSES)

gpu_load()
device = device_detection()

device = device_detection()
model = get_densNet_model()
model_state_dict = torch.load("model.pth", weights_only=True, map_location=device)
model.load_state_dict(model_state_dict)
model = model.to(device)


y_test = []
y_pred = []

fig = plt.figure(figsize=(10, 10))
class_labels = {0: "cat", 1: "dog"}

model.eval()
with torch.no_grad():
    for b, (X_test, y_test_temp) in enumerate(get_test_loader()):
        X_test = X_test.to(device)
        y_test_temp = y_test_temp.to(device)
        y_val = model(X_test)
        predicted = torch.sigmoid(y_val) > 0.5
        y_test.extend(y_test_temp.cpu().numpy().reshape(-1))
        y_pred.extend(predicted.cpu().numpy().reshape(-1))

acc = accuracy_score(y_test, y_pred)
print(f"accuracy is {acc * 100}%")
f1 = f1_score(y_test, y_pred, average="macro")  # important for multiclass
print(f"F1 score: {f1:.4f}")
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, xticklabels=CLASSES, yticklabels=CLASSES)
