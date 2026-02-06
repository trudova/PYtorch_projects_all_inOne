import numpy as np
import pandas as pd
import torch
from LoanClassificationNN import LoanClassificationNN
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from utils.device import device_detection, gpu_load

gpu_load()
device = device_detection()

data_pd = pd.read_csv("cleaned_data.csv")
data_pd_X = data_pd.drop(columns=["loan_status"])
data_np = data_pd_X.values.astype(np.float32)
actuale = data_pd["loan_status"].to_numpy(dtype=np.float32)
# make torch tensors for TensorDataset
data = torch.tensor(data_np, dtype=torch.float32)

labels = torch.tensor(actuale, dtype=torch.float32).unsqueeze(1)

test_dataloader = DataLoader(TensorDataset(data, labels), batch_size=32, shuffle=False)

model = LoanClassificationNN(num_features=data.shape[1]).to(device)
model.load_state_dict(torch.load("model_3.pth", map_location=device))

y_pred = []
model.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        inputs = X.to(device)
        labels = y.to(device)

        outputs = model(inputs)  # logits
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()

        y_pred.extend(predicted.cpu().numpy().reshape(-1).tolist())

accuracy = accuracy_score(actuale.reshape(-1), np.array(y_pred))
confusion = confusion_matrix(actuale.reshape(-1), np.array(y_pred))
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
