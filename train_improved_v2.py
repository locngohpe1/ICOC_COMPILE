"""
train_googlenet_optimal.py - OPTIMAL GOOGLENET TRAINING
========================================================

Key Changes from "Improved" Version:
1. ✅ DISABLED Mixup - Was too strong, caused underfitting
2. ✅ REDUCED Label Smoothing - 0.15 → 0.1
3. ✅ WEAKER Augmentation - Less aggressive transforms
4. ✅ EARLY STOPPING - patience=15 (no epoch waste!)
5. ✅ ReduceLROnPlateau - Proven to work better than Cosine
6. ✅ Moderate regularization - Allow model to fit better

Expected Results:
- Train Accuracy: 91-93% (vs 87.87% before)
- Val Accuracy: 90-91% (vs 89.80% before)
- Convergence: 30-40 epochs (vs 13 then plateau)
- Overfitting Gap: 2-3% (healthy, vs -1.93% underfitting)
- Training Time: ~25-30 minutes (vs 70 mins wasted)

This config balances:
- Enough regularization (prevent overfit)
- Not too much (allow fitting)
- Early stopping (save time)
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
from obstacle_classifier import ObstacleClassifier


def train_optimal():
    """
    Optimal training function for GoogLeNet
    """
    print("=" * 70)
    print("OPTIMAL GOOGLENET TRAINING")
    print("=" * 70)
    print("🎯 Configuration:")
    print("   - NO Mixup (allow better fitting)")
    print("   - Moderate Label Smoothing (0.1)")
    print("   - Balanced Augmentation")
    print("   - Early Stopping (patience=15)")
    print("   - ReduceLROnPlateau (proven effective)")
    print("   - Target: 90-91% accuracy in 30-40 epochs")
    print("=" * 70)

    # ==================== CHECK DATASET ====================
    train_dir = 'data/obstacles/train'
    val_dir = 'data/obstacles/val'

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("\n❌ ERROR: Dataset not found!")
        print("   Please run: python rearrange.py")
        return

    # ==================== BALANCED DATA AUGMENTATION ====================
    print("\n📊 Setting up BALANCED data augmentation...")

    # Training: MODERATE augmentation (not too weak, not too strong)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Balanced (was 0.7 in improved)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # Moderate (was 20 in improved)
        transforms.ColorJitter(
            brightness=0.2,  # Moderate (was 0.3 in improved)
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        # NO RandomAffine - removed for less regularization
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

    print("✅ Augmentation configured (BALANCED):")
    print("   - RandomResizedCrop: scale=(0.8, 1.0) - MODERATE")
    print("   - RandomRotation: 15° - MODERATE")
    print("   - ColorJitter: 0.2 strength - MODERATE")
    print("   - RandomAffine: DISABLED (less regularization)")

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

    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU REQUIRED!")

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
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    mem_free, mem_total = torch.cuda.mem_get_info(0)
    print(f"   GPU Memory Available: {mem_free / 1024**3:.1f} GB")

    # ==================== OPTIMAL TRAINING CONFIGURATION ====================
    print("\n" + "=" * 70)
    print("OPTIMAL TRAINING CONFIGURATION")
    print("=" * 70)

    max_epochs = 100  # Max epochs (will stop early if no improvement)
    initial_lr = 0.001
    use_mixup = False  # ← DISABLED!
    label_smoothing = 0.1  # ← REDUCED from 0.15
    weight_decay = 1e-4  # Standard weight decay
    early_stop_patience = 15  # Stop if no improvement for 15 epochs
    lr_patience = 5  # Reduce LR if no improvement for 5 epochs

    print(f"Max epochs: {max_epochs} (will stop early)")
    print(f"Initial learning rate: {initial_lr}")
    print(f"Optimizer: SGD with momentum (0.9)")
    print(f"Weight decay: {weight_decay} (standard)")
    print(f"")
    print(f"LR Scheduler: ReduceLROnPlateau")
    print(f"  - Mode: min (reduce when val_loss stops decreasing)")
    print(f"  - Factor: 0.5 (multiply LR by 0.5)")
    print(f"  - Patience: {lr_patience} epochs")
    print(f"  - Min LR: 1e-6")
    print(f"")
    print(f"Loss function: CrossEntropyLoss")
    print(f"  - Label smoothing: {label_smoothing} (REDUCED from 0.15)")
    print(f"")
    print(f"Mixup augmentation: ✗ DISABLED (was causing underfitting)")
    print(f"Gradient clipping: max_norm=1.0")
    print(f"")
    print(f"Early stopping: ✓ ENABLED")
    print(f"  - Patience: {early_stop_patience} epochs")
    print(f"  - Saves time and prevents overfitting!")

    print(f"\n📈 Expected Results:")
    print(f"  - Train Accuracy: 91-93% (vs 87.87% before)")
    print(f"  - Val Accuracy: 90-91% (vs 89.80% before)")
    print(f"  - Convergence: 30-40 epochs (vs 150 wasted)")
    print(f"  - Overfitting Gap: 2-3% (healthy)")
    print(f"  - Training Time: ~25-30 minutes (vs 70 mins)")

    print(f"\n🔑 Key Improvements:")
    print(f"  1. Less regularization → Better fitting")
    print(f"  2. Early stopping → No wasted epochs")
    print(f"  3. Proven LR schedule → Stable training")
    print(f"  4. Balanced approach → Best of both worlds")

    print("=" * 70)

    input("\n✋ Press ENTER to start OPTIMAL training or Ctrl+C to cancel... ")

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

    # Loss function with moderate label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    print(f"✅ Loss: CrossEntropyLoss (label_smoothing={label_smoothing})")

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=0.9,
        weight_decay=weight_decay
    )
    print(f"✅ Optimizer: SGD (lr={initial_lr}, momentum=0.9, wd={weight_decay})")

    # ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=lr_patience,
        min_lr=1e-6
    )
    print(f"✅ Scheduler: ReduceLROnPlateau (patience={lr_patience}, factor=0.5)")

    # Tracking variables
    best_val_acc = 0.0
    best_epoch = 0
    training_history = []
    patience_counter = 0

    print("\n" + "=" * 70)
    print(f"🚀 STARTING OPTIMAL TRAINING (MAX {max_epochs} EPOCHS)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: GoogLeNet (pretrained on ImageNet)")
    print(f"Early Stopping: patience={early_stop_patience}")
    print("=" * 70 + "\n")

    # ==================== TRAINING LOOP ====================
    for epoch in range(max_epochs):
        # ==================== TRAINING PHASE ====================
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # NO Mixup - train normally
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Handle GoogLeNet tuple output
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
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
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # ==================== MODEL CHECKPOINTING ====================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0  # Reset counter

            # Save best model
            if not os.path.exists('models'):
                os.makedirs('models')

            best_model_path = "models/googlenet_obstacle_classifier_optimal_best.pth"
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

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

        print(f"Epoch [{epoch + 1}/{max_epochs}] | LR: {new_lr:.6f} | Patience: {patience_counter}/{early_stop_patience}")
        print(f"  Train: Loss={epoch_loss:.4f} | Acc={train_acc:.4f} ({train_acc * 100:.2f}%)")
        print(f"  Val:   Loss={val_loss:.4f} | Acc={val_acc:.4f} ({val_acc * 100:.2f}%)")
        print(f"  Best:  {best_val_acc:.4f} ({best_val_acc * 100:.2f}%) at epoch {best_epoch}")
        print(f"  Gap:   {overfitting_gap:+.2f}%")

        # LR reduction notification
        if old_lr != new_lr:
            print(f"  📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

        # ==================== EARLY STOPPING CHECK ====================
        if patience_counter >= early_stop_patience:
            print(f"\n{'=' * 70}")
            print(f"⏹️  EARLY STOPPING TRIGGERED!")
            print(f"{'=' * 70}")
            print(f"   No improvement for {early_stop_patience} epochs")
            print(f"   Best validation accuracy: {best_val_acc * 100:.2f}% (epoch {best_epoch})")
            print(f"   Stopping at epoch {epoch + 1}")
            print(f"   Saved {max_epochs - (epoch + 1)} epochs (~{(max_epochs - (epoch + 1)) * 0.5:.0f} minutes)!")
            print(f"{'=' * 70}\n")
            break

        # Milestone logging (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            print(f"\n{'=' * 70}")
            print(f"🎯 MILESTONE: {epoch + 1}/{max_epochs} epochs completed")
            print(f"   Best val accuracy: {best_val_acc * 100:.2f}% (epoch {best_epoch})")
            print(f"   Current LR: {new_lr:.6f}")
            print(f"   Overfitting gap: {overfitting_gap:+.2f}%")
            if overfitting_gap < 0:
                print(f"   ⚠️  Model underfitting (val > train)")
            elif overfitting_gap < 3:
                print(f"   ✅ Healthy gap (good generalization)")
            elif overfitting_gap < 5:
                print(f"   ⚠️  Slight overfitting (acceptable)")
            else:
                print(f"   🚨 Overfitting detected (gap > 5%)")
            print(f"{'=' * 70}\n")
        else:
            print(f"{'-' * 70}")

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # ==================== TRAINING COMPLETED ====================
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 70)

    # Restore best model
    best_model_path = "models/googlenet_obstacle_classifier_optimal_best.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"✅ Restored best model (epoch {best_epoch}, accuracy: {best_val_acc * 100:.2f}%)")

    # Save final model
    final_model_path = "models/googlenet_obstacle_classifier_optimal.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved to: {final_model_path}")

    # Save training history
    df = pd.DataFrame(training_history)
    history_path = 'models/training_history_optimal.csv'
    df.to_csv(history_path, index=False)
    print(f"✅ Training history saved to: {history_path}")

    # ==================== FINAL SUMMARY ====================
    print(f"\n{'=' * 70}")
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total epochs trained: {len(training_history)}")
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}% (epoch {best_epoch})")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Compare with previous versions
    print(f"\n📈 Comparison with Previous Versions:")
    print(f"   Old model:      89.10% (epoch 48, 68 total)")
    print(f"   Improved model: 89.80% (epoch 13, 150 total - wasted 137 epochs)")
    print(f"   Optimal model:  {best_val_acc * 100:.2f}% (epoch {best_epoch}, {len(training_history)} total)")

    if best_val_acc > 0.898:
        improvement = (best_val_acc - 0.898) * 100
        print(f"   → Improvement: +{improvement:.2f}% vs Improved! 🎉")
    elif best_val_acc > 0.891:
        improvement = (best_val_acc - 0.891) * 100
        print(f"   → Improvement: +{improvement:.2f}% vs Old! 🎉")

    print(f"\nFiles saved:")
    print(f"  - Best model: {best_model_path}")
    print(f"  - Final model: {final_model_path}")
    print(f"  - Training history: {history_path}")

    print(f"\n🎯 Next steps:")
    print(f"  1. Run: python evaluate_model.py")
    print(f"  2. Compare confusion matrix with previous versions")
    print(f"  3. Check if Static→Dynamic errors improved")
    print("=" * 70)


if __name__ == "__main__":
    train_optimal()