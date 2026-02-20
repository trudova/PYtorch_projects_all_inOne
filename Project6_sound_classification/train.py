import torch
import torch.nn as nn
from Data_loading import get_test_loader, get_testset, get_train_loader, get_trainset
from matplotlib import pyplot as plt
from SoundCNNModel import SoundCNNModel
from utils.Device import device_detection

device = device_detection()
model = SoundCNNModel().to(device)

CLASSES = ["artifact", "extrahls", "extrastole", "murmur", "normal"]
NUM_CLASSES = len(CLASSES)
LR = 0.0009
loss_fc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

NUM_EPOCHS = 200
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for e in range(NUM_EPOCHS):
    trn_corr = 0

    model.train()
    for i, (X_train, y_train) in enumerate(get_train_loader()):
        optimizer.zero_grad()
        X_train = X_train.to(device)
        y_train = y_train.long()
        y_train = y_train.to(device)
        y_output = model(X_train)
        loss = loss_fc(y_output, y_train)
        loss.backward()
        optimizer.step()

        predicted = torch.max(y_output.data, dim=1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        if i % 50 == 0:
            print("TRAINING Loss", loss.item())
    train_losses.append(loss.item())
    train_correct.append(trn_corr.item())

    model.eval()
    tst_corr = 0
    with torch.no_grad():
        for i, (X_test, y_test) in enumerate(get_test_loader()):
            X_test = X_test.to(device)
            y_test = y_test.long().to(device)
            y_output = model(X_test)
            predicted = torch.max(y_output.data, dim=1)[1]
            batch_corr = (predicted == y_test).sum()
            tst_corr += batch_corr
            test_loss = loss_fc(y_output, y_test)
        print(f"TESTING LOSS {test_loss.item()}")
        test_losses.append(test_loss.item())
        test_correct.append(tst_corr.item())

    acc = tst_corr.item() * 100 / len(get_testset())

    if test_loss.item() < 0.005 and loss.item() < 0.005 and acc > 98:
        acc = tst_corr.item() * 100 / len(get_testset())
        torch.save(model.state_dict(), f"model_{e}_{acc}.pth")

    print(
        f"Epoch {e} - Training accuracy: {trn_corr.item() * 100 / len(get_trainset()):.2f}%, Test accuracy: {tst_corr.item() * 100 / len(get_testset()):.2f}%"
    )

plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Testing loss")
plt.legend()

plt.figure()
plt.plot([t / len(get_trainset()) for t in train_correct], label="Training accuracy")
plt.plot([t / len(get_testset()) for t in test_correct], label="Testing accuracy")
plt.legend()
