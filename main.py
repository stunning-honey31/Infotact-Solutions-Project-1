import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path

from models.model import CustomCNN

# Config
BATCH_SIZE = 256
EPOCHS = 50
LR = 0.001
IMAGE_SIZE = 128

# Paths
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"
MODEL_SAVE_PATH = "outputs/model.pth"


def validate_dataset_dirs(train_dir, val_dir, test_dir):
    missing = [p for p in [train_dir, val_dir, test_dir] if not os.path.isdir(p)]
    if missing:
        missing_text = "\n".join([f"- {m}" for m in missing])
        raise FileNotFoundError(
            "Dataset folders are missing. Expected these directories:\n"
            f"{missing_text}\n\n"
            "Create/populate them before running training."
        )

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Fail fast with a readable message instead of a deep torchvision traceback.
validate_dataset_dirs(TRAIN_DIR, VAL_DIR, TEST_DIR)

# Datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("Train classes:", train_dataset.classes)
print("Val classes:", val_dataset.classes)
print("Test classes:", test_dataset.classes)
print("Train batches:", len(train_loader))

num_classes = len(train_dataset.classes)

# Model
model = CustomCNN(num_classes=num_classes).to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Metrics storage
train_losses = []
val_accuracies = []

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    print(f"\ Starting Epoch {epoch+1}/{EPOCHS}")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Live progress
        if i % 10 == 0:
            print(f"Epoch {epoch+1} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} Completed | Total Loss: {total_loss:.4f}")

    # Validation and its accuracy
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # Save metrics AFTER computation
    train_losses.append(total_loss)
    val_accuracies.append(val_acc)

# Saving the model
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n Model saved at: {MODEL_SAVE_PATH}")

# Testing Evaluation
model.eval()
correct, total = 0, 0

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * correct / total
print(f" Final Test Accuracy: {test_acc:.2f}%")

# Saving metrics for further use - mostly visualizations, but will be decided if the rest agree
metrics = {
    "train_loss": train_losses,
    "val_accuracy": val_accuracies
}

with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f)

print(" Metrics saved!")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
np.save("outputs/confusion_matrix.npy", cm)

print(" Confusion matrix saved!")
