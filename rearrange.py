# rearrange.py - With image validation
import os
import shutil
import random
from glob import glob
from PIL import Image


def is_valid_image(path):
    """
    Kiểm tra ảnh có hợp lệ không
    """
    try:
        img = Image.open(path)
        img.verify()
        img.close()
        return True
    except:
        return False


def reorganize_final():
    """
    Reorganize OpenImages data with validation:
    - 10,000 train + 2,000 val static (Chair/Table/Couch/Bed)
    - 10,000 train + 2,000 val dynamic (Person/Dog/Cat)
    - Skip corrupt images automatically
    """
    random.seed(42)

    # Create directories
    for split in ['train', 'val']:
        for cls in ['static', 'dynamic']:
            os.makedirs(f'data/obstacles/{split}/{cls}', exist_ok=True)

    print("Collecting and validating images...")

    # Collect static images WITH VALIDATION
    static_images = []
    for cls in ['Chair', 'Table', 'Couch', 'Bed']:
        images = glob(f'data/obstacles_oi/{cls}/images/*.jpg')
        # Validate each image
        valid_images = [img for img in images if is_valid_image(img)]
        corrupt_count = len(images) - len(valid_images)
        if corrupt_count > 0:
            print(f"  ⚠️  {cls}: Found {corrupt_count} corrupt images (skipped)")
        static_images.extend(valid_images)

    # Collect dynamic images WITH VALIDATION
    dynamic_images = []
    for cls in ['Person', 'Dog', 'Cat']:
        images = glob(f'data/obstacles_oi/{cls}/images/*.jpg')
        # Validate each image
        valid_images = [img for img in images if is_valid_image(img)]
        corrupt_count = len(images) - len(valid_images)
        if corrupt_count > 0:
            print(f"  ⚠️  {cls}: Found {corrupt_count} corrupt images (skipped)")
        dynamic_images.extend(valid_images)

    print(f"\n✓ Static: {len(static_images)} valid images")
    print(f"✓ Dynamic: {len(dynamic_images)} valid images")

    # Check if enough images
    if len(static_images) < 12000:
        print(f"❌ ERROR: Not enough static images!")
        print(f"   Need: 12000, Have: {len(static_images)}")
        return

    if len(dynamic_images) < 12000:
        print(f"❌ ERROR: Not enough dynamic images!")
        print(f"   Need: 12000, Have: {len(dynamic_images)}")
        return

    # Shuffle and split
    random.shuffle(static_images)
    random.shuffle(dynamic_images)

    splits = {
        'static_train': static_images[:10000],
        'static_val': static_images[10000:12000],
        'dynamic_train': dynamic_images[:10000],
        'dynamic_val': dynamic_images[10000:12000]
    }

    # Copy files
    print("\nCopying files...")
    for name, files in splits.items():
        cls, split = name.split('_')
        for i, src in enumerate(files):
            dst = f'data/obstacles/{split}/{cls}/{cls}_{i:04d}.jpg'
            shutil.copy2(src, dst)
        print(f"  ✓ {name}: {len(files)} images")

    print(f"\n✓ Dataset ready: 20,000 train + 4,000 val")
    print(f"✓ All images validated - no corrupt files!")


if __name__ == "__main__":
    reorganize_final()