import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# devise detection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

df = pd.read_csv("cleaned_data.csv")

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    df.drop(columns=["loan_status"]).values.astype(np.float32),
    df["loan_status"].values.astype(np.float32).reshape(-1, 1),
    test_size=0.33,
    random_state=42,
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_np)
X_test_s = scaler.transform(X_test_np)  # ✅ important

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
y_test_t = torch.tensor(y_test_np, dtype=torch.float32)

train_data_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True
)
test_data_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False
)


class LoanClassificationNN(nn.Module):
    def __init__(self, num_features, hidden_features1=20, hidden_features2=10):
        super().__init__()
        self.layer1 = nn.Linear(num_features, hidden_features1)
        self.layer2 = nn.Linear(hidden_features1, hidden_features2)
        self.layer3 = nn.Linear(hidden_features2, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


model = LoanClassificationNN(num_features=X_train_t.shape[1]).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
train_losses = []
test_losses = []
train_correct = []
test_correct = []
final_predictions = []
for epoch in range(num_epochs):
    trn_corr = 0
    tst_corr = 0
    # --- training ---
    model.train()
    for b, (Xb, yb) in enumerate(train_data_loader):
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        y_pred = model(Xb)

        loss = loss_fn(y_pred, yb)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(y_pred)
        prediction = (probs > 0.5).float()
        batch_corr = (prediction == yb).sum().item()
        trn_corr += batch_corr
        if b % 200 == 0:
            print(f"Epoch {epoch} Batch {b} loss {loss.item():.2f}")
    train_losses.append(loss.item())
    train_correct.append(trn_corr)

    # --- testing ---

    model.eval()
    with torch.no_grad():
        for b, (Xb, yb) in enumerate(test_data_loader):
            Xb, yb = Xb.to(device), yb.to(device)
            y_val = model(Xb)
            loss = loss_fn(y_pred, yb)
            probs = torch.sigmoid(y_val)

            prediction = (probs > 0.5).float()

            batch_corr = (prediction == yb).sum().item()
            tst_corr += batch_corr
            loss = loss_fn(y_val, yb)

        test_losses.append(loss.item())
        test_correct.append(tst_corr)
    print(
        f"Epoch {epoch} - Training accuracy: {trn_corr * 100 / len(X_train_np):.2f}%, Test accuracy: {tst_corr * 100 / len(X_test_np):.2f}%"
    )

plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Testing loss")
plt.legend()

# plt.plot([t / len(train_correct) for t in train_correct], label="Training accuracy")
# plt.plot([t / len(test_correct) for t in test_correct], label="Testing accuracy")
# plt.legend()

y_pred = []  # make it a list (you were using y_pred earlier as a tensor)

model.eval()
with torch.no_grad():
    for i, data in enumerate(test_data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)  # logits on device
        probs = torch.sigmoid(outputs)  # convert logits -> probabilities
        predicted = (probs > 0.5).float()  # 0/1

        y_pred.extend(predicted.cpu().numpy().reshape(-1).tolist())

accuracy = accuracy_score(y_test_np.reshape(-1), np.array(y_pred))
confusion = confusion_matrix(y_test_np.reshape(-1), np.array(y_pred))
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
