# download_openimages_simple.py
import sys
sys.path.insert(0, '.')

from openimages_lib.download import download_images

# Download
download_images(
    dest_dir="data/obstacles_oi",
    class_labels=["Bed", "Chair", "Couch", "Table", "Person", "Dog", "Cat"],
    meta_dir="openimages_metadata",
    limit=20000  # 20000 per class
)