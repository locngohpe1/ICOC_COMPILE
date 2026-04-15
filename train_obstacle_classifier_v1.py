import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from collections import Counter
import os
from obstacle_classifier import ObstacleClassifier


def train_obstacle_classifier():
    """
    Main training function for obstacle classifier
    """
    print("=" * 70)
    print("OBSTACLE CLASSIFIER TRAINING")
    print("=" * 70)

    # ==================== CHECK DATASET ====================
    train_dir = 'data/obstacles/train'
    val_dir = 'data/obstacles/val'

    if not os.path.exists(train_dir):
        print("\n❌ ERROR: Training data not found!")
        print(f"   Expected: {train_dir}")
        print("\n   Please run:")
        print("   1. python download_openimages_simple.py")
        print("   2. python rearrange.py")
        return

    if not os.path.exists(val_dir):
        print("\n❌ ERROR: Validation data not found!")
        print(f"   Expected: {val_dir}")
        return

    # ==================== DATA AUGMENTATION ====================
    print("\n📊 Setting up data augmentation...")

    # Training: Strong augmentation to prevent overfitting
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation: No augmentation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==================== LOAD DATASETS ====================
    print("\n📦 Loading datasets...")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"✅ Training samples: {len(train_dataset)}")

    # ✅ FIX: Efficient counting without loading images
    train_counts = Counter([c for _, c in train_dataset.samples])
    train_static = train_counts[0]
    train_dynamic = train_counts[1]
    print(f"   - Static: {train_static}")
    print(f"   - Dynamic: {train_dynamic}")

    print(f"✅ Validation samples: {len(val_dataset)}")

    # ✅ FIX: Efficient counting without loading images
    val_counts = Counter([c for _, c in val_dataset.samples])
    val_static = val_counts[0]
    val_dynamic = val_counts[1]
    print(f"   - Static: {val_static}")
    print(f"   - Dynamic: {val_dynamic}")

    # ==================== CREATE DATA LOADERS ====================
    print("\n📊 Creating data loaders...")

    # ✅ FORCE GPU SETTINGS
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ GPU REQUIRED but CUDA is not available!\n"
            "   This training script requires GPU.\n"
            "   Please check your CUDA installation."
        )

    batch_size = 32
    # ✅ FIX: num_workers=0 on Windows to avoid multiprocessing errors
    # On Windows, multiprocessing with DataLoader often causes issues
    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"✅ Batch size: {batch_size}")
    print(f"✅ Num workers: {num_workers} (Windows compatibility)")
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")

    # ✅ PRINT GPU INFO
    print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"   GPU Memory Available: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.1f} GB")

    # ==================== TRAINING CONFIGURATION ====================
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    num_epochs = 100
    learning_rate = 0.001  # For pretrained weights
    use_pretrained = True  # Change to False for training from scratch

    print(f"Total epochs: {num_epochs}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Optimizer: SGD with momentum (0.9)")
    print(f"Weight decay: 1e-4")
    print(f"LR Scheduler: ReduceLROnPlateau")
    print(f"  - Mode: min (reduce when val_loss stops decreasing)")
    print(f"  - Factor: 0.5 (multiply LR by 0.5)")
    print(f"  - Patience: 5 epochs")
    print(f"  - Min LR: 1e-6")
    print(f"Loss function: CrossEntropyLoss (label_smoothing=0.1)")
    print(f"Gradient clipping: max_norm=1.0")
    print(f"Early stopping: 20 epochs patience")
    print(f"Data augmentation: ✓ Enabled (training only)")

    if use_pretrained:
        print(f"Pretrained weights: ✓ ImageNet (fine-tuning)")
    else:
        print(f"Pretrained weights: ✗ Training from scratch")
        learning_rate = 0.01  # Higher LR for training from scratch

    print(f"\nDataset: OpenImages V7")
    print(f"  - Training: 10,000 static + 10,000 dynamic = 20,000 images")
    print(f"  - Validation: 2,000 static + 2,000 dynamic = 4,000 images")
    print(f"  - Static classes: Chair, Table, Couch, Bed")
    print(f"  - Dynamic classes: Person, Dog, Cat")

    print(f"\nExpected training time with GPU: ~30-45 minutes")

    print("=" * 70)

    # ==================== WARNING ====================
    print("\n⚠️  IMPORTANT: Training will take ~30-45 minutes!")
    print("   Make sure:")
    print("   - Your computer won't go to sleep")
    print("   - You have stable power supply")
    print("   - You have at least 2GB free disk space")
    print("   - GPU drivers are up to date")
    print("\n" + "=" * 70)

    input("✋ Press ENTER to start training or Ctrl+C to cancel... ")

    # ==================== INITIALIZE MODEL ====================
    print("\n" + "=" * 70)
    print("🎯 INITIALIZING MODEL...")
    print("=" * 70)

    classifier = ObstacleClassifier(use_pretrained=use_pretrained, use_gpu=True)

    # ==================== START TRAINING ====================
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING...")
    print("=" * 70)

    history = classifier.train(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

    # ==================== TRAINING COMPLETE ====================
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Model saved to: models/googlenet_obstacle_classifier.pth")
    print(f"Best model saved to: models/googlenet_obstacle_classifier_best.pth")
    print(f"Training history: models/training_history.csv")
    print(f"\n🎯 Next step: python evaluate_model.py")
    print("=" * 70)


if __name__ == "__main__":
    train_obstacle_classifier()