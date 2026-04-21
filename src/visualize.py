import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_loss(losses):
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("outputs/plots/confusion_matrix.png")
    plt.show()

    return cm


def plot_class_accuracy(cm, classes):
    class_acc = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(10,5))
    plt.bar(classes, class_acc)
    plt.xticks(rotation=90)

    plt.title("Class-wise Accuracy")
    plt.ylabel("Accuracy")

    plt.savefig("outputs/plots/class_accuracy.png")
    plt.show()

def show_predictions(model, loader, classes):
    import torch
    model.eval()
    images, labels = next(iter(loader))
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    fig, axes = plt.subplots(1,5, figsize=(15,5))
    for i in range(5):
        axes[i].imshow(images[i].permute(1,2,0))
        axes[i].set_title(f"P:{classes[preds[i]]}\nT:{classes[labels[i]]}")
        axes[i].axis("off")
    plt.show()