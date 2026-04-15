import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from obstacle_classifier import ObstacleClassifier
import os


def evaluate_model():
    """
    Comprehensive model evaluation with confusion matrix
    """
    print("=" * 70)
    print("MODEL EVALUATION - CONFUSION MATRIX ANALYSIS")
    print("=" * 70)

    # Check if model exists
    model_path = "models/googlenet_obstacle_classifier_weighted.pth"
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model not found at {model_path}")
        print("   Please train the model first!")
        return

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("\n📦 Loading datasets...")
    val_dataset = datasets.ImageFolder('data/obstacles/val', transform=val_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"✅ Validation samples: {len(val_dataset)}")

    # Load model
    print("\n🎯 Loading best model...")
    classifier = ObstacleClassifier(use_pretrained=True, use_gpu=True)
    classifier.model.load_state_dict(torch.load(model_path, map_location=classifier.device))
    classifier.model.eval()
    print("✅ Model loaded successfully")

    # Evaluate
    print("\n📊 Running evaluation...")
    all_labels = []
    all_predictions = []
    all_confidences = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(classifier.device)
            labels = labels.to(classifier.device)

            outputs = classifier.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)

    # Calculate accuracy
    accuracy = (all_predictions == all_labels).mean()
    print(f"\n✅ Validation Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(all_labels, all_predictions)
    class_names = ['Static', 'Dynamic']

    print("\n     Predicted")
    print("         Static  Dynamic")
    print("Actual")
    print(f"Static    {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"Dynamic   {cm[1][0]:5d}   {cm[1][1]:5d}")

    # Per-class metrics
    print("\n" + "=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)

    for i, class_name in enumerate(class_names):
        # True Positives, False Positives, False Negatives
        tp = cm[i][i]
        fp = cm[1 - i][i]
        fn = cm[i][1 - i]
        tn = cm[1 - i][1 - i]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{class_name}:")
        print(f"  Precision: {precision:.4f} ({precision * 100:.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall * 100:.2f}%)")
        print(f"  F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
        print(f"  Support:   {cm[i].sum()}")

    # Misclassification analysis
    print("\n" + "=" * 70)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 70)

    static_as_dynamic = cm[0][1]
    dynamic_as_static = cm[1][0]
    total_errors = static_as_dynamic + dynamic_as_static

    print(f"\nTotal misclassifications: {total_errors} ({total_errors / len(all_labels) * 100:.2f}%)")
    print(f"\n  Static classified as Dynamic: {static_as_dynamic} ({static_as_dynamic / cm[0].sum() * 100:.2f}%)")
    print(f"  Dynamic classified as Static: {dynamic_as_static} ({dynamic_as_static / cm[1].sum() * 100:.2f}%)")
    print(f"the class is divied into {dynamic_as_static}")
    # Confidence analysis
    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)

    correct_mask = all_predictions == all_labels
    incorrect_mask = ~correct_mask

    correct_conf = all_confidences[correct_mask]
    incorrect_conf = all_confidences[incorrect_mask]

    print(f"\nCorrect predictions:")
    print(f"  Mean confidence: {correct_conf.mean():.4f} ({correct_conf.mean() * 100:.2f}%)")
    print(f"  Std confidence:  {correct_conf.std():.4f}")
    print(f"  Min confidence:  {correct_conf.min():.4f}")

    print(f"\nIncorrect predictions:")
    print(f"  Mean confidence: {incorrect_conf.mean():.4f} ({incorrect_conf.mean() * 100:.2f}%)")
    print(f"  Std confidence:  {incorrect_conf.std():.4f}")
    print(f"  Max confidence:  {incorrect_conf.max():.4f}")

    # Low confidence predictions
    low_conf_threshold = 0.6
    low_conf_mask = all_confidences < low_conf_threshold
    low_conf_count = low_conf_mask.sum()
    low_conf_accuracy = (
                all_predictions[low_conf_mask] == all_labels[low_conf_mask]).mean() if low_conf_count > 0 else 0

    print(f"\nLow confidence predictions (< {low_conf_threshold}):")
    print(f"  Count: {low_conf_count} ({low_conf_count / len(all_labels) * 100:.2f}%)")
    if low_conf_count > 0:
        print(f"  Accuracy: {low_conf_accuracy:.4f} ({low_conf_accuracy * 100:.2f}%)")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    issues = []

    if static_as_dynamic > dynamic_as_static * 1.2:
        issues.append("⚠️  Model tends to classify Static as Dynamic")
        issues.append("   → Consider: More static training data or balance loss weights")
    elif dynamic_as_static > static_as_dynamic * 1.2:
        issues.append("⚠️  Model tends to classify Dynamic as Static")
        issues.append("   → Consider: More dynamic training data or stronger augmentation")

    if incorrect_conf.mean() > 0.7:
        issues.append("⚠️  High confidence on incorrect predictions")
        issues.append("   → Consider: Increase label smoothing or add mixup")

    if low_conf_count > len(all_labels) * 0.1:
        issues.append("⚠️  Many low confidence predictions")
        issues.append("   → Consider: Longer training or different architecture")

    if len(issues) == 0:
        print("\n✅ No major issues detected!")
        print("   Model is performing well and balanced")
    else:
        for issue in issues:
            print(f"\n{issue}")

    # Save visualization
    print("\n" + "=" * 70)
    print("SAVING VISUALIZATION")
    print("=" * 70)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n✅ Confusion matrix saved to: results/confusion_matrix.png")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_model()