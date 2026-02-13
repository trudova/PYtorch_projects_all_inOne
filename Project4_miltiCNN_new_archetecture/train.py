import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Data import get_test_loader, get_testset, get_train_loader, get_trainset
from Model import get_model
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.Device import device_detection
from utils.transformer import transform

device = device_detection()
transform = transform()

trainset = get_trainset()
testset = get_testset()

train_loader = get_train_loader()
test_loader = get_test_loader()

CLASSES = ["affenpinscher", "akita", "corgi"]
NUM_CLASSES = len(CLASSES)

model = get_model()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0009)

NUM_EPOCHS = 12
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for e in range(NUM_EPOCHS):
    trn_corr = 0
    tst_corr = 0
    model.train()
    for i, (X_train, y_train) in enumerate(train_loader):
        optimizer.zero_grad()
        # y_train = F.one_hot(y_train, num_classes=10).type(torch.float32).to(device)
        y_train = y_train.long()
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        y_output = model(X_train)
        loss = loss_fn(y_output, y_train)
        loss.backward()
        optimizer.step()

        predicted = torch.max(y_output.data, dim=1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        if i % 100 == 0:
            print("Loss", loss.item())
    train_losses.append(loss.item())
    train_correct.append(trn_corr.item())
    # torch.save(model.state_dict(), f"model_{e}.pth")

    model.eval()
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            # y_test = y_test.float().reshape(-1, 1)
            y_test = y_test.long()
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_val = model(X_test)
            predicted = torch.max(y_val.data, dim=1)[1]
            batch_corr = (predicted == y_test).sum()
            tst_corr += batch_corr
        loss = loss_fn(y_val, y_test)
        test_losses.append(loss.item())
        test_correct.append(tst_corr.item())
    print(
        f"Epoch {e} - Training accuracy: {trn_corr.item() * 100 / len(trainset):.2f}%, Test accuracy: {tst_corr.item() * 100 / len(testset):.2f}%"
    )


plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Testing loss")
plt.legend()

plt.figure()
plt.plot([t / len(trainset) for t in train_correct], label="Training accuracy")
plt.plot([t / len(testset) for t in test_correct], label="Testing accuracy")
plt.legend()


y_test = []
y_pred = []

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
print(CLASSES)
print(cmx)

print("FINAL accuracy", acc * 100, "%")
