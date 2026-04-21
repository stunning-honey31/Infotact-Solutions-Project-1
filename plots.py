import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# Load metrics
with open("outputs/metrics.json", "r") as f:
    metrics = json.load(f)

train_loss = metrics["train_loss"]
val_acc = metrics["val_accuracy"]

# LOSS PLOT
plt.figure()
plt.plot(train_loss)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("outputs/loss.png")
plt.close()

# VALIDATION ACCURACY
plt.figure()
plt.plot(val_acc)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("outputs/val_accuracy.png")
plt.close()

print("Basic plots saved!")

#CONFUSION MATRIX
cm = np.load("outputs/confusion_matrix.npy")

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("Confusion matrix plotted!")