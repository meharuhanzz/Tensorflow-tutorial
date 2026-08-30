"""Generates a tiny on-disk image dataset so main.py has something real
to point image_dataset_from_directory at, without needing to download
anything. Creates three classes of small solid-colour-ish images with
random noise -- nothing fancy, just enough to load as real files.

Run this once before main.py:  python3 make_sample_images.py
"""
import os
import random

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "sample_images")

CLASSES = {
    "red_ish": (200, 30, 30),
    "green_ish": (30, 200, 30),
    "blue_ish": (30, 30, 200),
}
IMAGES_PER_CLASS = 8
SIZE = (64, 64)


def make_noisy_image(base_color):
    img = Image.new("RGB", SIZE, base_color)
    pixels = img.load()
    for x in range(SIZE[0]):
        for y in range(SIZE[1]):
            r, g, b = pixels[x, y]
            noise = random.randint(-40, 40)
            pixels[x, y] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise)),
            )
    return img


def main():
    random.seed(42)
    for class_name, color in CLASSES.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            img = make_noisy_image(color)
            img.save(os.path.join(class_dir, f"{class_name}_{i}.png"))
    print(f"Wrote {len(CLASSES) * IMAGES_PER_CLASS} images to {DATA_DIR}/")


if __name__ == "__main__":
    main()
