"""TensorFlow Day 8 -- Convolutional Neural Networks.

Run me with:  python3 main.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf


def conv_block(filters):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
    ])


# ---- 1. A small CNN: two conv blocks + Flatten + Dense head ----
print("=== CNN with Flatten head ===")
model = tf.keras.Sequential([
    conv_block(8),
    conv_block(16),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3),
])
x = tf.random.uniform((2, 32, 32, 3))
out = model(x)
print(f"output shape: {out.shape}")
model.summary()

# ---- 2. padding="same" vs "valid" ----
print("\n=== padding: same vs valid ===")
conv_same = tf.keras.layers.Conv2D(4, 3, padding="same")
conv_valid = tf.keras.layers.Conv2D(4, 3, padding="valid")
print(f"same:  input (32,32) -> output {conv_same(x).shape[1:3]}")
print(f"valid: input (32,32) -> output {conv_valid(x).shape[1:3]} (shrinks -- no padding added)")

# ---- 3. strides on Conv2D instead of a separate pooling layer ----
print("\n=== strides=2 vs MaxPooling2D(2) ===")
via_pool = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
])
via_stride = tf.keras.layers.Conv2D(8, 3, padding="same", strides=2, activation="relu")
print(f"via pooling layer: {via_pool(x).shape}")
print(f"via strides=2:     {via_stride(x).shape} (same spatial shrink, one fewer layer)")

# ---- 4. Flatten vs GlobalAveragePooling2D -- parameter count ----
print("\n=== Flatten vs GlobalAveragePooling2D ===")
model_flatten = tf.keras.Sequential([conv_block(8), conv_block(16), tf.keras.layers.Flatten(), tf.keras.layers.Dense(3)])
model_gap = tf.keras.Sequential([conv_block(8), conv_block(16), tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(3)])
model_flatten(x)
model_gap(x)
print(f"Flatten head params: {model_flatten.count_params()}")
print(f"GAP head params:     {model_gap.count_params()} (far fewer -- GAP collapses each")
print(f"feature map to one number before the Dense layer, instead of keeping all of it)")
