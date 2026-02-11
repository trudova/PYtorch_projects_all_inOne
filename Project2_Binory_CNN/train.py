import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

batch_size = 4

trainset = torchvision.datasets.ImageFolder(root="./data/train", transform=transform)
testset = torchvision.datasets.ImageFolder(root="./data/test", transform=transform)

train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)


class ImmageClassificationNet(nn.Module):
    def __init__(self):
        super(ImmageClassificationNet, self).__init__()
        # 1 color chanal, 6 filters(out channels), 3x3 kernel, stride(step) 1, padding 1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

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
        x = self.sigmoid(x)
        return x


model = ImmageClassificationNet()

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
    train_correct.append(trn_corr)

    model.eval()
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_test = y_test.float().reshape(-1, 1)
            y_val = model(X_test).round()
            batch_corr = (y_val == y_test).sum()
            tst_corr += batch_corr
        loss = loss_fn(y_val, y_test)
        test_losses.append(loss.item())
        test_correct.append(tst_corr)
    print(
        f"Epoch {epoch + 1} - Training accuracy: {trn_corr.item() * 100 / len(trainset):.2f}%, Test accuracy: {tst_corr.item() * 100 / len(testset):.2f}%"
    )


y_test = []
y_pred = []
model.eval()
with torch.no_grad():
    for b, (X_test, y_test_temp) in enumerate(test_loader):
        y_val = model(X_test).round()
        y_test.extend(y_test_temp.numpy())
        y_pred.extend(y_val.numpy())

acc = accuracy_score(y_test, y_pred)
print("accurasy", acc)
