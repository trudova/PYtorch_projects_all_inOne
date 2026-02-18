import torch
import torchvision
from Data import get_test_loader, get_testset, get_train_loader, get_trainset
from matplotlib import pyplot as plt
from Model import get_model
from tqdm import tqdm
from utils.Device import device_detection
from utils.transformer import transform

# detect device
device = device_detection()

# invoce transformer
transform = transform()
# get data
trainset = get_trainset()
testset = get_testset()
# get loaders
train_loader = get_train_loader()
test_loader = get_test_loader()

CLASSES = ["defective", "good"]
LEARNING_RATE = 0.0009

# creating the restNet model and cleaning the last output channel
resnet50_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)
resnet50_model.fc = torch.nn.Identity()  # remove the final classification layer

for param in resnet50_model.parameters():
    param.requires_grad = False  # freeze the pre-trained layers

resnet50_model.eval()
resnet50_model.to(device=device)

# getting custome model
fc_model = get_model()
# creating hybrid model
model = torch.nn.Sequential(resnet50_model, fc_model)
model = model.to(device)

# loss func and optimizer
loss_fc = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(fc_model.parameters(), lr=LEARNING_RATE)

NUM_EPOCHS = 10

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for e in range(NUM_EPOCHS):
    trn_corr = 0
    tst_corr = 0
    model.train()
    resnet50_model.eval()  # presearve gradient
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
    torch.save(fc_model.state_dict(), f"fc_model_{e}.pth")
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
        loss = loss_fc(y_val, y_test)
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
