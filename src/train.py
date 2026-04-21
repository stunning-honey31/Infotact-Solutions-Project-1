import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Allow running this file directly (python src/train.py) as well as module imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEVICE, EPOCHS, LR
import os

def train_model(model, train_loader):
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    use_amp = DEVICE.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            # Support both (images, labels) and (images, labels, paths) loaders.
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, labels = batch[0], batch[1]
            else:
                raise ValueError("Each batch must contain at least images and labels.")

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    # Keep both paths for compatibility with scripts/app that expect either location.
    torch.save(model.state_dict(), "outputs/model.pth")
    torch.save(model.state_dict(), "outputs/models/model.pth")
# torch.save(model.state_dict(), "outputs/models/model.pth")


def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.legend()
    plt.title("Loss Curve")

    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/loss_curve.png")
    plt.show()
    
if __name__ == "__main__":
    print("Training started...")

    from src.dataset import get_dataloaders
    from models.model import CustomCNN   

    train_loader, val_loader, class_names = get_dataloaders()

    model = CustomCNN(num_classes=len(class_names))

    train_model(model, train_loader)

    print("Training completed!")