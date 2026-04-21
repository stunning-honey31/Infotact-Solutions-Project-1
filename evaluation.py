import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from src.config import DEVICE


def evaluate_model(model, loader, class_names):

    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    all_confidences = []
    misclassified = []

    top3_correct = 0
    total = 0

    print("\n Running Evaluation...")

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            conf, preds = torch.max(probs, 1)
            _, top3 = torch.topk(outputs, 3, dim=1)

            for i in range(labels.size(0)):
                if labels[i] in top3[i]:
                    top3_correct += 1

                if preds[i] != labels[i]:
                    misclassified.append({
                        "path": paths[i],
                        "pred": preds[i].item(),
                        "true": labels[i].item()
                    })

            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(conf.cpu().numpy())

    # Metrics
    top1_acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    top3_acc = 100 * top3_correct / total

    print(f"\n Top-1 Accuracy: {top1_acc:.2f}%")
    print(f" Top-3 Accuracy: {top3_acc:.2f}%")

    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    os.makedirs("outputs", exist_ok=True)

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # =========================
    # PRECISION / RECALL / F1
    # =========================
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    precision = [report[c]['precision'] for c in class_names]
    recall = [report[c]['recall'] for c in class_names]
    f1 = [report[c]['f1-score'] for c in class_names]

    x = np.arange(len(class_names))

    plt.figure(figsize=(14, 6))
    plt.bar(x - 0.2, precision, 0.2, label='Precision')
    plt.bar(x, recall, 0.2, label='Recall')
    plt.bar(x + 0.2, f1, 0.2, label='F1-score')
    plt.xticks(x, class_names, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/class_metrics.png")
    plt.close()

    # =========================
    # CONFIDENCE DISTRIBUTION
    # =========================
    plt.figure()
    plt.hist(all_confidences, bins=20)
    plt.title("Confidence Distribution")
    plt.savefig("outputs/confidence_distribution.png")
    plt.close()

    # =========================
    # MISCLASSIFIED IMAGES
    # =========================
    misclassified = misclassified[:12]

    plt.figure(figsize=(12, 8))

    for i, item in enumerate(misclassified):
        img = Image.open(item["path"])

        plt.subplot(3, 4, i+1)
        plt.imshow(img)
        plt.axis("off")

        pred_name = class_names[item["pred"]].replace("_", " ")
        true_name = class_names[item["true"]].replace("_", " ")

        plt.title(f"P: {pred_name}\nT: {true_name}")

    plt.tight_layout()
    plt.savefig("outputs/misclassified.png")
    plt.close()

    print(" All plots saved in outputs/")
    
    
    
def plot_dataset_distribution(train_dir):
    import os
    import matplotlib.pyplot as plt

    class_counts = {}
    for class_name in os.listdir(train_dir):
        path = os.path.join(train_dir, class_name)
        if os.path.isdir(path):
            class_counts[class_name] = len(os.listdir(path))

    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=90)
    plt.title("Dataset Distribution")
    plt.tight_layout()
    plt.savefig("outputs/dataset_distribution.png")
    plt.close()

    print(" Dataset distribution plotted!")
    
    

def grad_cam(model, image_tensor):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    layer = list(model.children())[0][-1]

    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred = output.argmax()

    model.zero_grad()
    output[0, pred].backward()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1)

    cam = torch.relu(cam)
    cam = cam / cam.max()

    return cam.squeeze().cpu().numpy()



