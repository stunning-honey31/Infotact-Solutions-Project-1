from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.transforms import train_transform, val_transform
from src.config import TRAIN_DIR, VAL_DIR, BATCH_SIZE

def get_dataloaders():
    train_data = ImageFolder(TRAIN_DIR, transform=train_transform)
    val_data = ImageFolder(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, train_data.classes