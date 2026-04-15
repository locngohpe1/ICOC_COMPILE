"""
train_improved.py - IMPROVED OBSTACLE CLASSIFIER TRAINING
=========================================================

Key Improvements:
1. Cosine Annealing with Warm Restarts - Prevent early convergence
2. Mixup Augmentation - Better generalization & reduce overconfidence
3. Stronger Data Augmentation - Prevent overfitting
4. Reduced Weight Decay - Allow better fitting
5. Higher Label Smoothing - Better calibration
6. No Early Stopping - Train full 150 epochs
7. Better logging & checkpointing

Expected Results:
- Accuracy: 89% → 91-93%
- Static→Dynamic errors: 245 → 180-200
- Better confidence calibration
- More balanced performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from collections import Counter
import os
import pandas as pd
import gc
import numpy as np
from obstacle_classifier import ObstacleClassifier


def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation - Mix two samples together

    Args:
        x: Input batch [B, C, H, W]
        y: Labels [B]
        alpha: Mixup strength (0.2 recommended)

    Returns:
        mixed_x: Mixed inputs
        y_a, y_b: Original labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute mixup loss

    Args:
        criterion: Loss function
        pred: Model predictions
        y_a, y_b: Original labels
        lam: Mixing coefficient

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_improved():
    """
    Main improved training function
    """
    print("=" * 70)
    print("IMPROVED OBSTACLE CLASSIFIER TRAINING")
    print("=" * 70)
    print("🚀 Features:")
    print("   - Cosine Annealing with Warm Restarts")
    print("   - Mixup Augmentation (α=0.2)")
    print("   - Stronger Data Augmentation")
    print("   - Reduced Weight Decay (5e-5)")
    print("   - Label Smoothing (0.15)")
    print("   - Full 150 Epochs (No Early Stopping)")
    print("=" * 70)

    # ==================== CHECK DATASET ====================
    train_dir = 'data/obstacles/train'
    val_dir = 'data/obstacles/val'

    if not os.path.exists(train_dir):
        print("\n❌ ERROR: Training data not found!")
        print(f"   Expected: {train_dir}")
        print("   Please run: python rearrange.py")
        return

    if not os.path.exists(val_dir):
        print("\n❌ ERROR: Validation data not found!")
        print(f"   Expected: {val_dir}")
        return

    # ==================== STRONGER DATA AUGMENTATION ====================
    print("\n📊 Setting up STRONGER data augmentation...")

    # Training: STRONGER augmentation to prevent overfitting
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # More aggressive (was 0.8)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),  # Increased from 15
        transforms.ColorJitter(
            brightness=0.3,  # Increased from 0.2
            contrast=0.3,  # Increased from 0.2
            saturation=0.3,  # Increased from 0.2
            hue=0.15  # Increased from 0.1
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # NEW: Translation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation: No augmentation (standard)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("✅ Augmentation configured:")
    print("   - RandomResizedCrop: scale=(0.7, 1.0) - MORE aggressive")
    print("   - RandomRotation: 20° - INCREASED")
    print("   - ColorJitter: 0.3 strength - STRONGER")
    print("   - RandomAffine: translate=(0.1, 0.1) - NEW")

    # ==================== LOAD DATASETS ====================
    print("\n📦 Loading datasets...")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"✅ Training samples: {len(train_dataset)}")
    train_counts = Counter([c for _, c in train_dataset.samples])
    print(f"   - Static: {train_counts[0]}")
    print(f"   - Dynamic: {train_counts[1]}")

    print(f"✅ Validation samples: {len(val_dataset)}")
    val_counts = Counter([c for _, c in val_dataset.samples])
    print(f"   - Static: {val_counts[0]}")
    print(f"   - Dynamic: {val_counts[1]}")

    # ==================== CREATE DATA LOADERS ====================
    print("\n📊 Creating data loaders...")

    # Force GPU check
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ GPU REQUIRED but CUDA is not available!\n"
            "   This training script requires GPU.\n"
            "   Please check your CUDA installation."
        )

    batch_size = 32
    num_workers = 0  # Windows-safe

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

    print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    mem_free, mem_total = torch.cuda.mem_get_info(0)
    print(f"   GPU Memory Available: {mem_free / 1024 ** 3:.1f} GB")

    # ==================== TRAINING CONFIGURATION ====================
    print("\n" + "=" * 70)
    print("IMPROVED TRAINING CONFIGURATION")
    print("=" * 70)

    num_epochs = 150
    initial_lr = 0.002
    use_mixup = True
    mixup_alpha = 0.2
    label_smoothing = 0.15
    weight_decay = 5e-5
    T_0 = 30  # Cosine restart cycle
    T_mult = 2

    print(f"Total epochs: {num_epochs} (increased from 100)")
    print(f"Initial learning rate: {initial_lr} (increased from 0.001)")
    print(f"Optimizer: SGD with momentum (0.9)")
    print(f"Weight decay: {weight_decay} (REDUCED from 1e-4)")
    print(f"")
    print(f"LR Scheduler: CosineAnnealingWarmRestarts")
    print(f"  - T_0: {T_0} epochs (first cycle length)")
    print(f"  - T_mult: {T_mult} (cycle length multiplier)")
    print(f"  - eta_min: 1e-6 (minimum LR)")
    print(f"  → Prevents early convergence with warm restarts!")
    print(f"")
    print(f"Loss function: CrossEntropyLoss")
    print(f"  - Label smoothing: {label_smoothing} (increased from 0.1)")
    print(f"  → Better confidence calibration!")
    print(f"")
    print(f"Mixup augmentation: {'✓ Enabled' if use_mixup else '✗ Disabled'}")
    print(f"  - Alpha: {mixup_alpha}")
    print(f"  → Reduces overconfidence & bias!")
    print(f"")
    print(f"Gradient clipping: max_norm=1.0")
    print(f"Early stopping: ✗ DISABLED (train full {num_epochs} epochs)")

    print(f"\n📈 Expected Improvements:")
    print(f"  - Overall Accuracy: 89.1% → 91-93%")
    print(f"  - Static Recall: 87.75% → 90-92%")
    print(f"  - Static→Dynamic Errors: 245 → 180-200")
    print(f"  - Confidence Calibration: Better (72.79% → 65-68%)")
    print(f"  - Training Time: ~70 minutes")

    print("=" * 70)

    # Warning
    print("\n⚠️  IMPORTANT:")
    print("   - Training will take ~70 minutes")
    print("   - Accuracy will fluctuate (cosine schedule)")
    print("   - Best model may be at any epoch")
    print("   - No early stopping - full 150 epochs")
    print("\n" + "=" * 70)

    input("\n✋ Press ENTER to start IMPROVED training or Ctrl+C to cancel... ")

    # ==================== INITIALIZE MODEL ====================
    print("\n" + "=" * 70)
    print("🎯 INITIALIZING MODEL...")
    print("=" * 70)

    classifier = ObstacleClassifier(use_pretrained=True, use_gpu=True)
    device = classifier.device
    model = classifier.model

    # ==================== TRAINING SETUP ====================
    print("\n" + "=" * 70)
    print("🔧 SETTING UP TRAINING COMPONENTS...")
    print("=" * 70)

    # Loss function with stronger label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    print(f"✅ Loss: CrossEntropyLoss (label_smoothing={label_smoothing})")

    # Optimizer with reduced weight decay
    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=0.9,
        weight_decay=weight_decay
    )
    print(f"✅ Optimizer: SGD (lr={initial_lr}, momentum=0.9, wd={weight_decay})")

    # Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=1e-6
    )
    print(f"✅ Scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult})")

    # Tracking variables
    best_val_acc = 0.0
    best_epoch = 0
    training_history = []

    print("\n" + "=" * 70)
    print(f"🚀 STARTING IMPROVED TRAINING FOR {num_epochs} EPOCHS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: GoogLeNet (pretrained on ImageNet)")
    print(f"Classes: Static (Chair, Table, Couch, Bed) vs Dynamic (Person, Dog, Cat)")
    print("=" * 70 + "\n")

    # ==================== TRAINING LOOP ====================
    for epoch in range(num_epochs):
        # ==================== TRAINING PHASE ====================
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Apply Mixup augmentation
            if use_mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Handle GoogLeNet tuple output
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Compute loss (mixup or regular)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)

            if use_mixup:
                # For mixup, weighted accuracy
                total_train += labels_a.size(0)
                correct_train += (lam * (predicted == labels_a).float() +
                                  (1 - lam) * (predicted == labels_b).float()).sum().item()
            else:
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train

        # ==================== VALIDATION PHASE ====================
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                # Handle tuple output
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val

        # ==================== LEARNING RATE SCHEDULING ====================
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']

        # ==================== MODEL CHECKPOINTING ====================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            # Save best model
            if not os.path.exists('models'):
                os.makedirs('models')

            best_model_path = "models/googlenet_obstacle_classifier_improved_best.pth"
            torch.save(model.state_dict(), best_model_path)

        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': new_lr
        })

        # ==================== LOGGING ====================
        overfitting_gap = (train_acc - val_acc) * 100

        print(f"Epoch [{epoch + 1}/{num_epochs}] | LR: {new_lr:.6f}")
        print(f"  Train: Loss={epoch_loss:.4f} | Acc={train_acc:.4f} ({train_acc * 100:.2f}%)")
        print(f"  Val:   Loss={val_loss:.4f} | Acc={val_acc:.4f} ({val_acc * 100:.2f}%)")
        print(f"  Best:  {best_val_acc:.4f} ({best_val_acc * 100:.2f}%) at epoch {best_epoch}")
        print(f"  Gap:   {overfitting_gap:.2f}%")

        # Detect warm restart
        if new_lr > old_lr * 1.5:
            print(f"  🔄 WARM RESTART DETECTED: LR jumped from {old_lr:.6f} to {new_lr:.6f}")

        # Milestone logging (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            print(f"\n{'=' * 70}")
            print(f"🎯 MILESTONE: {epoch + 1}/{num_epochs} epochs completed")
            print(f"   Progress: {(epoch + 1) / num_epochs * 100:.1f}%")
            print(f"   Best val accuracy: {best_val_acc * 100:.2f}% (epoch {best_epoch})")
            print(f"   Current LR: {new_lr:.6f}")
            print(f"   Overfitting gap: {overfitting_gap:.2f}%")

            # Time estimate
            epochs_remaining = num_epochs - (epoch + 1)
            time_per_epoch = 28  # seconds (approximate)
            time_remaining = epochs_remaining * time_per_epoch / 60
            print(f"   Est. time remaining: ~{time_remaining:.0f} minutes")
            print(f"{'=' * 70}\n")
        else:
            print(f"{'-' * 70}")

        # ==================== MEMORY CLEANUP ====================
        gc.collect()
        torch.cuda.empty_cache()

    # ==================== TRAINING COMPLETED ====================
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 70)

    # Restore best model
    best_model_path = "models/googlenet_obstacle_classifier_improved_best.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"✅ Restored best model (epoch {best_epoch}, accuracy: {best_val_acc * 100:.2f}%)")

    # Save final model
    if not os.path.exists('models'):
        os.makedirs('models')

    final_model_path = "models/googlenet_obstacle_classifier_improved.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved to: {final_model_path}")

    # Save training history
    df = pd.DataFrame(training_history)
    history_path = 'models/training_history_improved.csv'
    df.to_csv(history_path, index=False)
    print(f"✅ Training history saved to: {history_path}")

    # ==================== FINAL SUMMARY ====================
    print(f"\n{'=' * 70}")
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total epochs trained: {len(training_history)}")
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}% (epoch {best_epoch})")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"")
    print(f"Files saved:")
    print(f"  - Best model: {best_model_path}")
    print(f"  - Final model: {final_model_path}")
    print(f"  - Training history: {history_path}")
    print(f"")
    print(f"🎯 Next steps:")
    print(f"  1. Run: python evaluate_model.py")
    print(f"  2. Compare with old model (89.1% → {best_val_acc * 100:.2f}%)")
    print(f"  3. Check confusion matrix improvements")
    print("=" * 70)


if __name__ == "__main__":
    train_improved()