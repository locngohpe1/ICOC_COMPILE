# train_v2.py - Training V2 with regularization (SIMPLE VERSION)
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from obstacle_classifier_v3 import ObstacleClassifierV3


def train_v2():
    """
    Train V2 with stronger regularization
    """
    print("=" * 70)
    print("TRAINING V3 - WITH REGULARIZATION")
    print("=" * 70)

    train_dir = 'data/obstacles/train'
    val_dir = 'data/obstacles/val'

    if not os.path.exists(train_dir):
        print("\n❌ ERROR: Training data not found!")
        return

    if not os.path.exists(val_dir):
        print("\n❌ ERROR: Validation data not found!")
        return

    # STRONGER AUGMENTATION
    print("\n📊 Setting up STRONGER augmentation...")

    # Moderate augmentation:
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Không quá aggressive
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # Giảm từ 20
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Bỏ RandomErasing, RandomGrayscale
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n📦 Loading datasets...")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"✅ Training: {len(train_dataset)}")
    print(f"✅ Validation: {len(val_dataset)}")

    print("\n📊 Creating data loaders...")

    batch_size = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"✅ Batch size: {batch_size}")
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")

    if torch.cuda.is_available():
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 70)
    print("REGULARIZATION TECHNIQUES:")
    print("  ✅ Dropout: 0.5")
    print("  ✅ Weight decay: 5e-4 (vs 1e-4)")
    print("  ✅ Label smoothing: 0.1")
    print("  ✅ Gradient clipping: 1.0")
    print("  ✅ Stronger augmentation")
    print("=" * 70)

    input("\n✋ Press ENTER to start or Ctrl+C to cancel... ")

    print("\n" + "=" * 70)
    print("🎯 INITIALIZING MODEL V3...")
    print("=" * 70)

    classifier = ObstacleClassifierV3(use_pretrained=True, use_gpu=True)

    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING V3...")
    print("=" * 70)

    history = classifier.train(
        train_loader,
        val_loader,
        num_epochs=100,
        learning_rate=0.001
    )

    print("\n" + "=" * 70)
    print("✅ TRAINING V3 COMPLETED!")
    print("=" * 70)
    print("Model: models/googlenet_obstacle_classifier_v3.pth")
    print("History: models/training_history_v3.csv")
    print("\n📊 Compare:")
    print("  V1 (baseline):    91.50%")
    print("  V2 (regularized): Check evaluate_v3.py")
    print("=" * 70)


if __name__ == "__main__":
    train_v2()