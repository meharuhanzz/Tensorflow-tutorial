"""TensorFlow Day 1 -- Tensors.

Run me with:  python3 main.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet the startup noise

import numpy as np
import tensorflow as tf

# ---- 1. Creating tensors ----
print("=== Creating tensors ===")
a = tf.constant([1, 2, 3])
print(f"a = {a}, dtype = {a.dtype}")

b = tf.constant([1.0, 2.0, 3.0])
print(f"b = {b}, dtype = {b.dtype}")

zeros = tf.zeros((2, 3))
ones = tf.ones((2, 3))
rand = tf.random.uniform((2, 3))
print(f"\nzeros:\n{zeros}")
print(f"ones:\n{ones}")
print(f"rand:\n{rand}")

# ---- 2. Shape, dtype, and dimensions ----
print("\n=== Shape & dtype ===")
matrix = tf.random.uniform((3, 4))
print(f"matrix.shape = {matrix.shape}")
print(f"matrix.dtype = {matrix.dtype}")
print(f"rank (ndim)  = {len(matrix.shape)}")

# ---- 3. Tensor operations ----
print("\n=== Operations ===")
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([4.0, 5.0, 6.0])
print(f"x + y = {x + y}")
print(f"x * y = {x * y}")
print(f"tensordot(x, y) = {tf.tensordot(x, y, axes=1)}")
print(f"tf.reduce_mean(x) = {tf.reduce_mean(x)}")
print(f"tf.reduce_sum(x) = {tf.reduce_sum(x)}")

m1 = tf.random.uniform((2, 3))
m2 = tf.random.uniform((3, 4))
result = m1 @ m2
print(f"\n(2x3) @ (3x4) -> shape {result.shape}")

# ---- 4. Indexing and slicing ----
print("\n=== Indexing & slicing ===")
t = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"t =\n{t}")
print(f"t[0] = {t[0]}")
print(f"t[:, 0] = {t[:, 0]}")
print(f"t[1, 2] = {t[1, 2]}")
print(f"t[0:2, 1:] = \n{t[0:2, 1:]}")

# ---- 5. Reshaping ----
print("\n=== Reshaping ===")
flat = tf.range(12)
print(f"flat = {flat}")
reshaped = tf.reshape(flat, (3, 4))
print(f"reshaped (3x4) =\n{reshaped}")
reshaped2 = tf.reshape(flat, (2, -1))
print(f"reshaped (2, -1) =\n{reshaped2}")

# ---- 6. Converting between tensors and NumPy ----
print("\n=== Tensor <-> NumPy ===")
np_array = np.array([1, 2, 3])
from_numpy = tf.constant(np_array)
print(f"tf.constant({np_array}) = {from_numpy}")

back_to_numpy = from_numpy.numpy()
print(f"back to numpy: {back_to_numpy}, type: {type(back_to_numpy)}")

# ---- 7. tf.Variable -- the mutable tensor used for model parameters ----
print("\n=== tf.Variable ===")
w = tf.Variable([1.0, 2.0, 3.0])
print(f"w before: {w.numpy()}")
w.assign(w * 2)
print(f"w after assign(w * 2): {w.numpy()}")

# ---- 8. Devices (GPU-ready code) ----
print("\n=== Device ===")
gpus = tf.config.list_physical_devices("GPU")
device = "/GPU:0" if gpus else "/CPU:0"
print(f"GPUs visible: {len(gpus)} -- using device: {device}")

with tf.device(device):
    on_device = tf.random.uniform((2, 2))
print(f"tensor placed via tf.device({device})")
