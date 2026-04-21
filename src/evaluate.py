import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from src.config import DEVICE

def evaluate_model(model, loader, class_names):
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    all_confidences = []

    top3_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            probs = F.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)

            # Top-3 predictions
            _, top3 = torch.topk(outputs, 3, dim=1)

            for i in range(labels.size(0)):
                if labels[i] in top3[i]:
                    top3_correct += 1

            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(conf.cpu().numpy())

    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    top3_acc = 100 * top3_correct / total

    print(f"\nTop-1 Accuracy: {accuracy * 100:.2f}%")
    print(f"🔥 Top-3 Accuracy: {top3_acc:.2f}%")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Save outputs
    np.save("outputs/preds.npy", np.array(all_preds))
    np.save("outputs/labels.npy", np.array(all_labels))
    np.save("outputs/confidences.npy", np.array(all_confidences))

    print("Predictions, labels, confidences saved!")

    return all_labels, all_preds

if __name__ == "__main__":
    print("Evaluation started...")

    from src.dataset import get_dataloaders
    from models.model import CustomCNN 

    _, val_loader, class_names = get_dataloaders()

    model = CustomCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("outputs/model.pth"))

    evaluate_model(model, val_loader, class_names)