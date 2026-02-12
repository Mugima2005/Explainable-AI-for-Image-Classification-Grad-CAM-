from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# datasets
train_data = datasets.ImageFolder("dataset/train", transform=transform)
test_data  = datasets.ImageFolder("dataset/test", transform=transform)

# dataloaders
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=16, shuffle=False)

print(train_data.classes)
