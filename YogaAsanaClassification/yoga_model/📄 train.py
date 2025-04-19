import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from yoga_model.dataset import YogaDataset
from yoga_model.model import YogaCNN
import torch.optim as optim
import torch.nn as nn
import os

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = YogaDataset("processed_data/final_dataset/train", transform=train_transforms)
val_dataset = YogaDataset("processed_data/final_dataset/val", transform=train_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = YogaCNN(num_classes=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")