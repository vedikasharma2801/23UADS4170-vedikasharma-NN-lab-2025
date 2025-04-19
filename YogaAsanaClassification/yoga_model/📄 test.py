# yoga_model/test.py

from yoga_model.dataset import YogaDataset
from yoga_model.model import YogaCNN
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

model = YogaCNN(num_classes=6)
model.load_state_dict(torch.load("yoga_cnn.pth"))
model.eval()

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = YogaDataset("processed_data/final_dataset/test", transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=16)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
