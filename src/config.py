import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 128
BATCH_SIZE = 128   # increased for GPU
EPOCHS = 30
LR = 0.001

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"