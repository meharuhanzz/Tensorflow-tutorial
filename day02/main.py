"""TensorFlow Day 2 -- tf.GradientTape (automatic differentiation).

Run me with:  python3 main.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

# ---- 1. tf.Variable -- the thing GradientTape tracks automatically ----
print("=== tf.Variable + GradientTape ===")
x = tf.Variable(3.0)
print(f"x = {x.numpy()}")

# ---- 2. Building a computation and calling tape.gradient() ----
with tf.GradientTape() as tape:
    y = x ** 2
print(f"y = x**2 = {y.numpy()}")

dy_dx = tape.gradient(y, x)
print(f"dy/dx (should be 2x = 6.0): {dy_dx.numpy()}")

# ---- 3. A slightly bigger example, by hand vs. autograd ----
print("\n=== Bigger example ===")
a = tf.Variable(2.0)
b = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = a ** 2 + b ** 3

dz_da, dz_db = tape.gradient(z, [a, b])
print(f"z = a**2 + b**3 = {z.numpy()}")
print(f"dz/da (should be 2a = 4.0): {dz_da.numpy()}")
print(f"dz/db (should be 3b^2 = 27.0): {dz_db.numpy()}")

# ---- 4. tape.watch() -- tracking a plain constant instead of a Variable ----
print("\n=== tape.watch() on a tf.constant ===")
c = tf.constant(4.0)
with tf.GradientTape() as tape:
    tape.watch(c)
    y = c ** 2
print(f"dy/dc (should be 2c = 8.0): {tape.gradient(y, c).numpy()}")

# ---- 5. The #1 early gotcha: a tape is single-use ----
print("\n=== Single-use tapes vs. persistent=True ===")
w = tf.Variable(5.0)
with tf.GradientTape() as tape:
    loss = w ** 2
grad1 = tape.gradient(loss, w)
print(f"first tape.gradient() call: {grad1.numpy()}")
try:
    tape.gradient(loss, w)
except RuntimeError as e:
    print(f"second call on the same (non-persistent) tape failed: {type(e).__name__}")

with tf.GradientTape(persistent=True) as tape:
    loss = w ** 2
    loss2 = w ** 3
print(f"persistent tape, 1st gradient: {tape.gradient(loss, w).numpy()}")
print(f"persistent tape, 2nd gradient (different output): {tape.gradient(loss2, w).numpy()}")
del tape

# ---- 6. No torch.no_grad() needed -- just don't open a tape ----
print("\n=== Plain ops build no graph at all ===")
x2 = tf.Variable(4.0)
y2 = x2 ** 2
print(f"y2 = {y2.numpy()} (computed with no tape -- cheaper, and that's the default)")

# ---- 7. tf.stop_gradient() -- PyTorch's .detach() ----
print("\n=== tf.stop_gradient() ===")
x3 = tf.Variable(4.0)
with tf.GradientTape() as tape:
    y3 = x3 ** 2
    y3_stopped = tf.stop_gradient(y3)
    total = y3_stopped * x3
grad = tape.gradient(total, x3)
print(f"grad only sees the x3 factor, not the y3_stopped one: {grad.numpy()}")

# ---- 8. A tiny "manual gradient descent" step -- the core idea behind training ----
print("\n=== One manual gradient descent step ===")
w = tf.Variable(0.0)
learning_rate = 0.1

for step in range(5):
    with tf.GradientTape() as tape:
        loss = (w - 10) ** 2
    grad = tape.gradient(loss, w)
    w.assign_sub(learning_rate * grad)
    print(f"step {step}: w = {w.numpy():.3f}, loss = {loss.numpy():.3f}")
