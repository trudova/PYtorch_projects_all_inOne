import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from Data import get_test_loader, get_testset, get_train_loader, get_trainset
from Model import get_model
from utils.Device import device_detection
from utils.transformer import transform

device = device_detection()
transform = transform()

trainset = get_trainset()
testset = get_testset()

train_loader = get_train_loader()
test_loader = get_test_loader()

CLASSES = ["affenpinscher", "akita", "corgi"]


resnet50_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)
resnet50_model.fc = torch.nn.Identity()  # remove the final classification layer

resnet50_model = resnet50_model.to(device)

fc_model = get_model()
fc_model = fc_model.to(device)

model = torch.nn.Sequential(resnet50_model, fc_model)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fc_model.parameters(), lr=0.0008)

NUM_EPOCHS = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for e in range(NUM_EPOCHS):
    trn_corr = 0
    tst_corr = 0
    model.train()
    resnet50_model.eval()
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
    torch.save(fc_model.state_dict(), f"fc_model_{e}.pth")
    train_losses.append(loss.item())
    train_correct.append(trn_corr.item())

    model.eval()
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
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
