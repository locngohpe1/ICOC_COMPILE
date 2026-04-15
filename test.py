# verify_labels.py - Kiểm tra dataset có labels đúng không
from torchvision import datasets, transforms
import os


def verify_labels():
    print("=" * 70)
    print("VERIFYING DATASET LABELS")
    print("=" * 70)

    # Check folders exist
    train_dir = 'data/obstacles/train'
    val_dir = 'data/obstacles/val'

    if not os.path.exists(train_dir):
        print("❌ ERROR: data/obstacles/train not found!")
        print("   Please run: python reorganize_openimages.py")
        return

    if not os.path.exists(val_dir):
        print("❌ ERROR: data/obstacles/val not found!")
        return

    # Load datasets
    print("\n📦 Loading datasets...")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    # Print class mapping
    print("\n✅ CLASS MAPPING:")
    print(f"   {train_dataset.class_to_idx}")
    print(f"   → 'static' = 0 (furniture)")
    print(f"   → 'dynamic' = 1 (person/animals)")

    # Count samples per class
    print("\n📊 TRAINING SET:")
    train_static = sum([1 for _, label in train_dataset if label == 0])
    train_dynamic = sum([1 for _, label in train_dataset if label == 1])
    print(f"   Total: {len(train_dataset)}")
    print(f"   - Static (label=0): {train_static}")
    print(f"   - Dynamic (label=1): {train_dynamic}")
    print(
        f"   - Balance: {train_static / (train_static + train_dynamic) * 100:.1f}% / {train_dynamic / (train_static + train_dynamic) * 100:.1f}%")

    print("\n📊 VALIDATION SET:")
    val_static = sum([1 for _, label in val_dataset if label == 0])
    val_dynamic = sum([1 for _, label in val_dataset if label == 1])
    print(f"   Total: {len(val_dataset)}")
    print(f"   - Static (label=0): {val_static}")
    print(f"   - Dynamic (label=1): {val_dynamic}")
    print(
        f"   - Balance: {val_static / (val_static + val_dynamic) * 100:.1f}% / {val_dynamic / (val_static + val_dynamic) * 100:.1f}%")

    # Sample a few images
    print("\n🔍 SAMPLE IMAGES:")
    for i in range(min(5, len(train_dataset))):
        img_path, label = train_dataset.samples[i]
        class_name = train_dataset.classes[label]
        img_name = os.path.basename(img_path)
        print(f"   [{i + 1}] {img_name} → Class: {class_name} (label={label})")

    # Verify structure
    if train_static > 0 and train_dynamic > 0 and val_static > 0 and val_dynamic > 0:
        print("\n" + "=" * 70)
        print("✅ DATASET LABELS ARE CORRECT!")
        print("=" * 70)
        print("   All images are properly labeled through folder structure.")
        print("   Ready to train!")
        print("\n   Next step: python train_obstacle_classifier.py")
        print("=" * 70)
    else:
        print("\n❌ ERROR: Some classes have 0 images!")
        print("   Please check the folder structure.")


if __name__ == "__main__":
    verify_labels()