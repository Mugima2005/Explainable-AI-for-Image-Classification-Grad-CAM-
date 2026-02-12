import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# -----------------------
# 1. DEVICE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# 2. TRANSFORMS
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------
# 3. DATASET & DATALOADER
# -----------------------
train_data = datasets.ImageFolder("dataset/train", transform=transform)
test_data  = datasets.ImageFolder("dataset/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=16, shuffle=False)

print("Classes:", train_data.classes)

# -----------------------
# 4. MODEL (ResNet18)
# -----------------------
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# change final layer for 2 classes
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# -----------------------
# 5. LOSS & OPTIMIZER
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# -----------------------
# 6. TRAINING LOOP
# -----------------------
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

# -----------------------
# 7. EVALUATION
# -----------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# -----------------------
# 8. SAVE MODEL
# -----------------------
torch.save(model.state_dict(), "resnet_crop.pth")
print("Model saved as resnet_crop.pth ✅")
