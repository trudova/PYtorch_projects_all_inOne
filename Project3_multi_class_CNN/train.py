import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomInvert(),
        transforms.RandomRotation(10),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

batch_size = 4
trainset = torchvision.datasets.ImageFolder(root="./data/train", transform=transform)
testset = torchvision.datasets.ImageFolder(root="./data/test", transform=transform)

train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

CLASSES = ["affenpinscher", "akita", "corgi"]
NUM_CLASSES = len(CLASSES)


class ImageMultiClassCNN(nn.Module):
    def __init__(self):
        super(ImageMultiClassCNN, self).__init__()
        #  color chanal, 6 filters(out channels), 3x3 kernel, stride(step) 1, padding 1
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.conv2 = nn.Conv2d(6, 16, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 1, 1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax(x)
        return x


model = ImageMultiClassCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

NUM_EPOCHS = 10
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
        y_train = y_train.long()
        # X_train = X_train.to(device)
        # y_train = y_train.to(device)
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
            # X_test = X_test.to(device)
            # y_test = y_test.to(device)
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
        # X_test = X_test.to(device)
        # y_test_temp = y_test_temp.to(device)
        y_val = model(X_test)  # logits shape [batch, 3]
        predicted = torch.argmax(y_val, dim=1)  # class ids shape [batch]

        y_test.extend(y_test_temp.cpu().numpy().reshape(-1))
        y_pred.extend(predicted.cpu().numpy().reshape(-1))

acc = accuracy_score(y_test, y_pred)
print("FINAL accuracy", acc * 100, "%")

cmx = confusion_matrix(y_test, y_pred)
print(CLASSES)
print(cmx)
