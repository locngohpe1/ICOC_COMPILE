# obstacle_classifier_v2.py - Simple version with dropout
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


class ObstacleClassifierV2:
    def __init__(self, use_pretrained=True, use_gpu=True):
        """
        GoogLeNet with Dropout (p=0.5)
        """
        self.device = torch.device('cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu')

        if use_pretrained:
            print("✅ Loading GoogLeNet with ImageNet pretrained weights + Dropout")
            self.model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        else:
            print("⚠️  Training from SCRATCH + Dropout")
            self.model = models.googlenet(weights=None, init_weights=True)

        self.model.aux1 = None
        self.model.aux2 = None

        # ADD DROPOUT
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 2)
        )

        model_path = "models/googlenet_obstacle_classifier_v2.pth"
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Loaded model from {model_path}")
            except:
                print(f"⚠️  Starting fresh")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = ['static', 'dynamic']
        print(f"✅ Model with Dropout initialized on {self.device}")

    def classify(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)

        return self.classes[predicted.item()], confidence.item()

    def train(self, train_dataloader, val_dataloader, num_epochs=100, learning_rate=0.001):
        """
        Train with stronger regularization
        """
        # Label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Higher weight decay
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4  # Increased
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
        )

        best_val_acc = 0.0
        training_history = []
        patience_counter = 0
        early_stop_patience = 30

        print(f"\n{'=' * 70}")
        print(f"TRAINING WITH REGULARIZATION:")
        print(f"  - Dropout: 0.5")
        print(f"  - Weight decay: 5e-4")
        print(f"  - Label smoothing: 0.1")
        print(f"  - Gradient clipping: 1.0")
        print(f"{'=' * 70}\n")

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in train_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_dataloader.dataset)
            train_acc = correct_train / total_train

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_dataloader.dataset)
            val_acc = correct_val / total_val

            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if not os.path.exists('models'):
                    os.makedirs('models')
                torch.save(self.model.state_dict(), "models/googlenet_obstacle_classifier_v2_best.pth")
            else:
                patience_counter += 1

            training_history.append({
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': new_lr
            })

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | LR: {new_lr:.6f} | Patience: {patience_counter}/{early_stop_patience}")
            print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc * 100:.2f}%)")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} ({val_acc * 100:.2f}%)")
            print(f"  Best Acc:   {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")

            if old_lr != new_lr:
                print(f"  📉 LR reduced: {old_lr:.6f} → {new_lr:.6f}")

            if patience_counter >= early_stop_patience:
                print(f"\n{'=' * 70}")
                print(f"⚠️  EARLY STOPPING at epoch {epoch + 1}")
                print(f"{'=' * 70}\n")
                break

            if (epoch + 1) % 25 == 0:
                print(f"\n{'=' * 70}")
                print(f"🎯 MILESTONE: {epoch + 1}/{num_epochs} epochs")
                print(f"   Best: {best_val_acc * 100:.2f}%")
                print(f"   Gap: {(train_acc - val_acc) * 100:.2f}%")
                print(f"{'=' * 70}\n")
            else:
                print(f"{'-' * 70}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Restore best
        best_path = "models/googlenet_obstacle_classifier_v2_best.pth"
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            print(f"\n✅ Restored best model ({best_val_acc * 100:.2f}%)")

        # Save
        torch.save(self.model.state_dict(), "models/googlenet_obstacle_classifier_v2.pth")
        print(f"✅ Saved to: models/googlenet_obstacle_classifier_v2.pth")

        df = pd.DataFrame(training_history)
        df.to_csv('models/training_history_v2.csv', index=False)
        print(f"✅ History saved")

        print(f"\n{'=' * 70}")
        print(f"🎉 TRAINING COMPLETED!")
        print(f"Best: {best_val_acc * 100:.2f}%")
        print(f"{'=' * 70}")

        return training_history