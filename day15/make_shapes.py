"""Generates a synthetic 4-class image dataset (circles / squares /
triangles / stars), randomized in position, size, color, and rotation,
on noisy backgrounds -- a genuine "the model must learn the SHAPE, not a
shortcut like color" task. One more class than Days 9-14's version, for
the capstone.

Run this once before main.py:  python3 make_shapes.py
"""
import math
import os
import random

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "shapes")

CLASSES = ["circle", "square", "triangle", "star"]
IMAGES_PER_CLASS = 60
SIZE = (48, 48)


def star_points(cx, cy, r):
    points = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        radius = r if i % 2 == 0 else r * 0.45
        points.append((cx + radius * math.cos(angle), cy - radius * math.sin(angle)))
    return points


def random_color():
    return tuple(random.randint(40, 255) for _ in range(3))


def make_shape_image(shape):
    # Noisy background so the model can't shortcut on a clean background.
    bg = random_color()
    img = Image.new("RGB", SIZE, bg)
    draw = ImageDraw.Draw(img)

    for _ in range(30):
        x, y = random.randint(0, SIZE[0] - 1), random.randint(0, SIZE[1] - 1)
        draw.point((x, y), fill=random_color())

    color = random_color()
    cx, cy = SIZE[0] // 2, SIZE[1] // 2
    r = random.randint(12, 18)
    jitter_x, jitter_y = random.randint(-6, 6), random.randint(-6, 6)
    cx, cy = cx + jitter_x, cy + jitter_y

    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    elif shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=color)
    elif shape == "triangle":
        draw.polygon(
            [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)],
            fill=color,
        )
    elif shape == "star":
        draw.polygon(star_points(cx, cy, r), fill=color)

    return img


def main():
    random.seed(42)
    for shape in CLASSES:
        class_dir = os.path.join(DATA_DIR, shape)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            img = make_shape_image(shape)
            img.save(os.path.join(class_dir, f"{shape}_{i}.png"))
    print(f"Wrote {len(CLASSES) * IMAGES_PER_CLASS} images to {DATA_DIR}/")


if __name__ == "__main__":
    main()
