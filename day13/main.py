"""TensorFlow Day 13 -- GPU & Mixed-Precision Training.

Run me with:  python3 main.py

Note: the real SPEEDUP from mixed precision is GPU-specific (tensor
cores). On CPU (very likely what's running this), every pattern below is
still 100% correct and will run -- you just won't see the performance
benefit until you run the same code on suitable GPU hardware.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

# ---- 1. Device check ----
print("=== Device ===")
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs visible: {len(gpus)}")

# ---- 2. Setting a global mixed-precision policy ----
print("\n=== mixed_bfloat16 policy ===")
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10),
])
x = tf.random.uniform((4, 20))
out = model(x)
print(f"layer dtype_policy: {model.layers[0].dtype_policy}")
print(f"model storage dtype (weights, still float32): {model.dtype}")
print(f"layer COMPUTE output dtype: {out.dtype}")

# ---- 3. mixed_float16 needs loss scaling -- compile() handles it for you ----
print("\n=== mixed_float16 + automatic LossScaleOptimizer ===")
tf.keras.mixed_precision.set_global_policy("mixed_float16")

model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(2),
])
model2.compile(optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
print(f"optimizer type after compile(): {type(model2.optimizer).__name__}")
print("(automatically wrapped in LossScaleOptimizer -- you'd only wrap it")
print(" yourself if writing a custom GradientTape training loop)")

# ---- 4. Comparing training with float32 vs mixed_bfloat16 ----
print("\n=== Comparing float32 vs mixed_bfloat16 on the same task ===")


def make_data():
    tf.random.set_seed(0)
    X = tf.random.normal((200, 10))
    Y = tf.cast(tf.reduce_sum(X[:, :3], axis=1) > 0, tf.int32)
    return X, Y


def train_with_policy(policy_name, epochs=10):
    tf.keras.mixed_precision.set_global_policy(policy_name)
    X, Y = make_data()
    m = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(2),
    ])
    m.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    history = m.fit(X, Y, epochs=epochs, verbose=0)
    return history.history["loss"][-1], history.history["accuracy"][-1]

loss_f32, acc_f32 = train_with_policy("float32")
loss_bf16, acc_bf16 = train_with_policy("mixed_bfloat16")
print(f"float32:        final loss = {loss_f32:.4f}, accuracy = {acc_f32:.3f}")
print(f"mixed_bfloat16: final loss = {loss_bf16:.4f}, accuracy = {acc_bf16:.3f}")
print("(should be close either way -- reduced precision costs speed/memory")
print(" benefits on GPU, not accuracy)")

tf.keras.mixed_precision.set_global_policy("float32")   # reset for cleanliness
