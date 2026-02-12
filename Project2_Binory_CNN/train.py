import torch
import torch.nn as nn
import torchvision
from ImmageClassificationNet import ImmageClassificationNet
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utils.device import device_detection
from utils.transformer import transform

device = device_detection()

transform = transform()

batch_size = 4

trainset = torchvision.datasets.ImageFolder(root="./data/train", transform=transform)
testset = torchvision.datasets.ImageFolder(root="./data/test", transform=transform)

train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)


model = ImmageClassificationNet()
model = model.to(device)

loss_fn = nn.BCELoss()
optimozer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

NUM_EPOCHS = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []
for epoch in range(NUM_EPOCHS):
    trn_corr = 0
    tst_corr = 0
    model.train()
    for i, (X_train, y_train) in enumerate(train_loader):
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        optimozer.zero_grad()
        y_output = model(X_train)

        loss = loss_fn(y_output, y_train)
        loss.backward()
        optimozer.step()

        prediction = (y_output > 0.5).float()
        batch_corr = (prediction == y_train).sum()
        trn_corr += batch_corr

        if i % 100 == 0:
            print("Loss", loss.item())
    train_losses.append(loss.item())
    train_correct.append(trn_corr.item())
    torch.save(model.state_dict(), f"model_{epoch}.pth")

    model.eval()
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_test = y_test.float().reshape(-1, 1)
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_val = model(X_test).round()
            batch_corr = (y_val == y_test).sum()
            tst_corr += batch_corr
        loss = loss_fn(y_val, y_test)
        test_losses.append(loss.item())
        test_correct.append(tst_corr.item())
    print(
        f"Epoch {epoch + 1} - Training accuracy: {trn_corr.item() * 100 / len(trainset):.2f}%, Test accuracy: {tst_corr.item() * 100 / len(testset):.2f}%"
    )


plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Testing loss")
plt.legend()

plt.figure()
plt.plot([t / len(trainset) for t in train_correct], label="Training accuracy")
plt.plot([t / len(testset) for t in test_correct], label="Testing accuracy")
plt.legend()
