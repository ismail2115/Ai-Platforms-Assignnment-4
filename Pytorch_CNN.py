# ======================================================
#      ADVANCED FASHION-MNIST CNN (PYTORCH)
# ======================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 1) Data Transform + Augmentation
# ===============================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor()
])

test_transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.FashionMNIST("./data", train=True, download=True, transform=train_transform)
test_data  = datasets.FashionMNIST("./data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

# ===============================
# 2) Advanced CNN
# ===============================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*3*3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

model = CNN().to(device)

# ===============================
# 3) Optimizer + Scheduler
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

# ===============================
# 4) Training Functions
# ===============================
def train_epoch():
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def test_epoch():
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss_sum += criterion(preds, y).item()

            _, predicted = torch.max(preds, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return loss_sum / len(test_loader), correct / total

# ===============================
# 5) Run Training
# ===============================
train_losses, test_losses, test_accs = [], [], []

for epoch in range(20):
    train_loss = train_epoch()
    test_loss, test_acc = test_epoch()
    
    scheduler.step(test_loss)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/20 | Train Loss={train_loss:.4f} | Test Loss={test_loss:.4f} | Acc={test_acc:.4f}")

# ===============================
# 6) Confusion Matrix + Report
# ===============================
all_preds, all_labels = [], []

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        preds = torch.argmax(model(x), 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.show()

print(classification_report(all_labels, all_preds))

# ===============================
# 7) Prediction Visualization
# ===============================
def show_predictions():
    imgs, labels = next(iter(test_loader))
    imgs = imgs[:12].to(device)

    preds = model(imgs)
    preds = torch.argmax(preds, 1).cpu()

    imgs = imgs.cpu()

    plt.figure(figsize=(12,6))
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(imgs[i].squeeze(), cmap="gray")
        plt.title(f"True={labels[i]} / Pred={preds[i]}")
        plt.axis("off")
show_predictions()

# ===============================
# 8) Grad-CAM
# ===============================
def grad_cam(img, model, layer):
    img = img.unsqueeze(0).to(device)
    model.eval()

    activations = []
    gradients = []

    def hook_activation(module, inp, out):
        activations.append(out)

    def hook_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = layer.register_forward_hook(hook_activation)
    h2 = layer.register_backward_hook(hook_gradient)

    preds = model(img)
    class_idx = preds.argmax()
    loss = preds[0, class_idx]

    model.zero_grad()
    loss.backward()

    heatmap = gradients[0].mean(dim=1)[0].cpu()
    heatmap = torch.relu(heatmap)
    heatmap /= heatmap.max() + 1e-8

    h1.remove()
    h2.remove()

    return heatmap.numpy()

sample_img, _ = test_data[0]
heatmap = grad_cam(sample_img, model, model.conv[0])

plt.imshow(sample_img.squeeze(), cmap="gray")
plt.imshow(heatmap, cmap="jet", alpha=0.5)
plt.show()

# ===============================
# 9) Save / Load Model
# ===============================
torch.save(model.state_dict(), "pytorch_fashion_mnist_advanced.pth")

loaded = CNN().to(device)
loaded.load_state_dict(torch.load("pytorch_fashion_mnist_advanced.pth"))
loaded.eval()

print("Loaded Model Test Accuracy:", test_epoch()[1])
