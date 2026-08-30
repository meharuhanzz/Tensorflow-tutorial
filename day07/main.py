"""TensorFlow Day 7 -- Real Image Data.

Run `python3 make_sample_images.py` once first, then:  python3 main.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "sample_images")

if not os.path.isdir(DATA_DIR):
    raise SystemExit("Run `python3 make_sample_images.py` first to generate sample_images/.")

# ---- 1. image_dataset_from_directory ----
print("=== Loading from folders ===")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, image_size=(32, 32), batch_size=4, shuffle=True, seed=42,
)
print(f"classes found: {train_ds.class_names}")
for images, labels in train_ds.take(1):
    print(f"one batch: images {images.shape}, labels {labels.numpy()}")

# ---- 2. Rescaling ----
print("\n=== Rescaling to [0, 1] ===")
rescale = tf.keras.layers.Rescaling(1.0 / 255)
raw_batch = next(iter(train_ds))[0]
print(f"before rescale: min={raw_batch.numpy().min():.1f}, max={raw_batch.numpy().max():.1f}")
rescaled_batch = rescale(raw_batch)
print(f"after rescale:  min={rescaled_batch.numpy().min():.3f}, max={rescaled_batch.numpy().max():.3f}")

# ---- 3. Data augmentation as model layers ----
print("\n=== Augmentation layers -- training=True vs False ===")
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
])

one_image = raw_batch[:1]
aug1 = augmentation(one_image, training=True)
aug2 = augmentation(one_image, training=True)
same_twice = np.allclose(aug1.numpy(), aug2.numpy())
print(f"two training=True calls identical? {same_twice} (should be False -- that's the randomness)")

no_aug1 = augmentation(one_image, training=False)
no_aug2 = augmentation(one_image, training=False)
unchanged = np.allclose(no_aug1.numpy(), one_image.numpy()) and np.allclose(no_aug2.numpy(), one_image.numpy())
print(f"two training=False calls both equal the original? {unchanged}")

# ---- 4. Attaching augmentation to the front of a model ----
print("\n=== Augmentation attached to a model ===")
model = tf.keras.Sequential([
    augmentation,
    tf.keras.layers.Rescaling(1.0 / 255),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3),
])
out_train = model(one_image, training=True)
out_eval_1 = model(one_image, training=False)
out_eval_2 = model(one_image, training=False)
print(f"model(x, training=False) deterministic across calls? "
      f"{np.allclose(out_eval_1.numpy(), out_eval_2.numpy())}")

# ---- 5. preprocess_input for pretrained models (preview of Day 11) ----
print("\n=== preprocess_input (ImageNet-pretrained-model preprocessing) ===")
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pp_mobilenet
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_resnet

sample_pixel = tf.constant([[[[100.0, 150.0, 200.0]]]])
print(f"raw pixel:              {sample_pixel.numpy().ravel()}")
print(f"mobilenet_v2 preprocess: {pp_mobilenet(tf.identity(sample_pixel)).numpy().ravel()}")
print(f"resnet50 preprocess:     {pp_resnet(tf.identity(sample_pixel)).numpy().ravel()}")
print("(different pretrained models expect different preprocessing -- always")
print(" use the one paired with the specific model you load, see Day 11)")
