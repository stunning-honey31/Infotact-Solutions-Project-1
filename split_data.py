import os
import shutil
import random

# Paths
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

TEST_SPLIT = 0.15   # 15% of data goes to test

# Create test folder
os.makedirs(TEST_DIR, exist_ok=True)

# Get all classes (from train)
classes = os.listdir(TRAIN_DIR)

for class_name in classes:
    train_class_path = os.path.join(TRAIN_DIR, class_name)
    val_class_path = os.path.join(VAL_DIR, class_name)
    test_class_path = os.path.join(TEST_DIR, class_name)

    os.makedirs(test_class_path, exist_ok=True)

    all_images = []

    # Collect from train
    if os.path.exists(train_class_path):
        for file in os.listdir(train_class_path):
            path = os.path.join(train_class_path, file)
            if os.path.isfile(path):
                all_images.append(path)

    # Collect from val
    if os.path.exists(val_class_path):
        for file in os.listdir(val_class_path):
            path = os.path.join(val_class_path, file)
            if os.path.isfile(path):
                all_images.append(path)

    # Shuffle
    random.shuffle(all_images)

    # Select test samples
    test_count = int(len(all_images) * TEST_SPLIT)
    test_samples = all_images[:test_count]

    # Copy to test folder
    for src in test_samples:
        filename = os.path.basename(src)
        dst = os.path.join(test_class_path, filename)
        shutil.copy(src, dst)

print(" Test dataset created successfully!")