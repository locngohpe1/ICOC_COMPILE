# obstacle_classifier.py - GoogLeNet Obstacle Classifier (FINAL CLEAN VERSION)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import GoogLeNet_Weights
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
import gc


class ObstacleClassifier:
    def __init__(self, use_pretrained=True, use_gpu=True):
        """
        Initialize GoogLeNet obstacle classifier

        Args:
            use_pretrained: Use ImageNet pretrained weights (recommended: True)
            use_gpu: Use GPU if available
        """
        # ✅ FORCE GPU CHECK
        if use_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "❌ GPU REQUIRED but CUDA is not available!\n"
                "   Please check:\n"
                "   1. NVIDIA GPU is installed\n"
                "   2. CUDA drivers are installed\n"
                "   3. PyTorch with CUDA support is installed\n"
                "   Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )

        # Device configuration - FORCE GPU
        self.device = torch.device('cuda:0')

        # Initialize model
        if use_pretrained:
            print("✅ Loading GoogLeNet with ImageNet pretrained weights")
            self.model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        else:
            print("⚠️  Training from SCRATCH (no pretrained weights)")
            self.model = models.googlenet(weights=None, init_weights=True)

        # Disable auxiliary classifiers for inference
        self.model.aux1 = None
        self.model.aux2 = None

        # Replace final FC layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

        # Proper weight initialization for new FC layer
        nn.init.kaiming_normal_(self.model.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.model.fc.bias, 0)
        print("✅ FC layer initialized with Kaiming normal")

        # Load existing model if available
        model_path = "models/googlenet_obstacle_classifier.pth"
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Loaded trained model from {model_path}")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                print(f"⚠️  Starting with fresh initialization...")

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Class names
        self.classes = ['static', 'dynamic']

        # ✅ PRINT GPU INFO
        print(f"✅ Model initialized on {self.device}")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    def classify(self, image):
        """
        Classify a single image as static or dynamic obstacle

        Args:
            image: PIL Image or numpy array

        Returns:
            class_name: 'static' or 'dynamic'
            confidence: float (0-1)
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Preprocess
        img = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img)

            # Handle potential tuple output from GoogLeNet
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)

        class_name = self.classes[predicted.item()]
        confidence = confidence.item()

        return class_name, confidence

    def classify_batch(self, images):
        """
        Classify multiple images at once

        Args:
            images: list of PIL Images or numpy arrays

        Returns:
            list of (class_name, confidence) tuples
        """
        # Prepare batch
        batch = []
        for image in images:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            batch.append(self.transform(image))

        batch_tensor = torch.stack(batch).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

        # Format results
        results = []
        for pred, conf in zip(predicted.cpu().numpy(), confidences.cpu().numpy()):
            results.append((self.classes[pred], float(conf)))

        return results

    def train(self, train_dataloader, val_dataloader, num_epochs=100, learning_rate=0.001):
        """
        Train the model on obstacle classification dataset

        Args:
            train_dataloader: PyTorch DataLoader for training
            val_dataloader: PyTorch DataLoader for validation
            num_epochs: number of training epochs
            learning_rate: initial learning rate (0.001 for pretrained, 0.01 for scratch)

        Returns:
            training_history: list of dicts with training metrics per epoch
        """
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimizer: SGD with momentum
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        # Tracking variables
        best_val_acc = 0.0
        training_history = []
        patience_counter = 0
        early_stop_patience = 20

        print(f"\n{'=' * 70}")
        print(f"STARTING TRAINING FOR {num_epochs} EPOCHS")
        print(f"{'=' * 70}")
        print(f"Initial learning rate: {learning_rate}")
        print(f"Optimizer: SGD (momentum=0.9, weight_decay=1e-4)")
        print(f"LR Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
        print(f"Early Stopping: {early_stop_patience} epochs patience")
        print(f"Loss function: CrossEntropyLoss (label_smoothing=0.1)")
        print(f"Gradient clipping: max_norm=1.0")
        print(f"Device: {self.device}")
        print(f"{'=' * 70}\n")

        for epoch in range(num_epochs):
            # ==================== TRAINING PHASE ====================
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in train_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Handle GoogLeNet tuple output
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_dataloader.dataset)
            train_acc = correct_train / total_train

            # ==================== VALIDATION PHASE ====================
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)

                    # Handle tuple output
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_dataloader.dataset)
            val_acc = correct_val / total_val

            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            # ==================== MODEL CHECKPOINTING ====================
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                # Save best model
                if not os.path.exists('models'):
                    os.makedirs('models')

                best_model_path = "models/googlenet_obstacle_classifier_best.pth"
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            # Save history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': new_lr
            })

            # ==================== LOGGING ====================
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                f"LR: {new_lr:.6f} | "
                f"Patience: {patience_counter}/{early_stop_patience}"
            )
            print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc * 100:.2f}%)")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} ({val_acc * 100:.2f}%)")
            print(f"  Best Acc:   {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")

            if old_lr != new_lr:
                print(f"  📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

            # ==================== EARLY STOPPING ====================
            if patience_counter >= early_stop_patience:
                print(f"\n{'=' * 70}")
                print(f"⚠️  EARLY STOPPING TRIGGERED!")
                print(f"{'=' * 70}")
                print(f"   No improvement for {early_stop_patience} epochs")
                print(f"   Best validation accuracy: {best_val_acc * 100:.2f}%")
                print(f"   Stopping at epoch {epoch + 1}")
                print(f"{'=' * 70}\n")
                break

            # Milestone logging
            if (epoch + 1) % 10 == 0:
                overfitting_gap = (train_acc - val_acc) * 100
                print(f"\n{'=' * 70}")
                print(f"🎯 MILESTONE: {epoch + 1}/{num_epochs} epochs completed")
                print(f"   Progress: {(epoch + 1) / num_epochs * 100:.1f}%")
                print(f"   Best val accuracy: {best_val_acc * 100:.2f}%")
                print(f"   Current LR: {new_lr:.6f}")
                print(f"   Overfitting gap: {overfitting_gap:.2f}%")
                print(f"{'=' * 70}\n")
            else:
                print(f"{'-' * 70}")

            # ==================== MEMORY CLEANUP ====================
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ==================== RESTORE BEST MODEL ====================
        best_model_path = "models/googlenet_obstacle_classifier_best.pth"
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"\n✅ Restored best model from disk (accuracy: {best_val_acc * 100:.2f}%)")

        # Save final model
        if not os.path.exists('models'):
            os.makedirs('models')

        final_model_path = "models/googlenet_obstacle_classifier.pth"
        torch.save(self.model.state_dict(), final_model_path)
        print(f"✅ Final model saved to: {final_model_path}")

        # Save training history
        df = pd.DataFrame(training_history)
        history_path = 'models/training_history.csv'
        df.to_csv(history_path, index=False)
        print(f"✅ Training history saved to: {history_path}")

        # ==================== FINAL SUMMARY ====================
        print(f"\n{'=' * 70}")
        print(f"🎉 TRAINING COMPLETED!")
        print(f"{'=' * 70}")
        print(f"Total epochs trained: {len(training_history)}")
        print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
        print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Model saved to: {final_model_path}")
        print(f"{'=' * 70}")

        return training_history