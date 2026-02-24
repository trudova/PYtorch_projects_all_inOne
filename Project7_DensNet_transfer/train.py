import torch
from Data import get_test_loader, get_testset, get_train_loader, get_trainset
from matplotlib import pyplot as plt
from Model import get_densNet_model
from tqdm import tqdm
from utils.Device import device_detection
from utils.transformer import get_transform

# detect device
device = device_detection()

# invoce transformer
transform = get_transform()
# get data
trainset = get_trainset()
testset = get_testset()
# get loaders
train_loader = get_train_loader()
test_loader = get_test_loader()


CLASSES = ["cat", "dog"]
LEARNING_RATE = 0.0005

model = get_densNet_model()
model.to(device)

# loss func and optimizer
loss_fc = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

NUM_EPOCHS = 20

train_losses = []
test_losses = []
train_correct = []
test_correct = []


for e in range(NUM_EPOCHS):
    trn_corr = 0
    tst_corr = 0
    # presearve gradient
    for i, (X_train, y_train) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        y_train = y_train.type(torch.float32).reshape(-1, 1)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        y_output = model(X_train)
        loss = loss_fc(y_output.to(device), y_train)
        loss.backward()
        optimizer.step()
        predicted = torch.sigmoid(y_output) > 0.5
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        if i % 100 == 0:
            print("Loss", loss.item())

    train_losses.append(loss.item())
    train_correct.append(trn_corr.item())

    model.eval()
    with torch.no_grad():
        for X_test, y_test in tqdm(test_loader):
            y_test = y_test.type(torch.float32).reshape(-1, 1)
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_val = model(X_test)
            predicted = torch.sigmoid(y_val) > 0.5
            batch_corr = (predicted == y_test).sum()
            tst_corr += batch_corr
        test_loss = loss_fc(y_val, y_test)
        test_losses.append(test_loss.item())
        test_correct.append(tst_corr.item())
        print(f"TESTING LOSS {test_loss.item()}")
    acc = tst_corr.item() * 100 / len(get_testset())

    if test_loss.item() < 0.05 and loss.item() < 0.05 and acc > 95:
        acc = tst_corr.item() * 100 / len(get_testset())
        torch.save(model.state_dict(), f"model_{e}_{acc}.pth")
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
