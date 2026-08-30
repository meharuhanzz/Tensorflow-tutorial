"""TensorFlow Day 3 -- Building a Model with tf.keras.Model.

Run me with:  python3 main.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf


# ---- 1. tf.keras.layers.Dense -- lazy building ----
print("=== Dense layer, lazy building ===")
layer = tf.keras.layers.Dense(units=1)
x = tf.random.uniform((5, 3))
output = layer(x)   # weights are created HERE, on first call
print(f"output shape: {output.shape}")
print(f"weights created: {[w.shape for w in layer.weights]}")


# ---- 2. Subclassing tf.keras.Model -- the standard pattern ----
print("\n=== Subclassing tf.keras.Model ===")
class SimpleNet(tf.keras.Model):
    def __init__(self):
        super().__init__()          # always call this first
        self.layer1 = tf.keras.layers.Dense(8)
        self.layer2 = tf.keras.layers.Dense(1)
        self.activation = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


model = SimpleNet()
print("summary() BEFORE the model has ever seen input (shows 0 params -- 'unbuilt'):")
model.summary()

out = model(tf.random.uniform((5, 4)))
print(f"\noutput shape: {out.shape}")
print("summary() AFTER one call -- now it knows every shape:")
model.summary()

# ---- 3. Inspecting a model ----
print("\n=== Inspecting ===")
print(f"count_params(): {model.count_params()}")
print(f"trainable_variables: {len(model.trainable_variables)} tensors")
for v in model.trainable_variables:
    print(f"  {v.name}: {v.shape}")

# ---- 4. tf.keras.Sequential -- a shortcut for simple pipelines ----
print("\n=== Sequential ===")
seq_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(2),
])
seq_model(tf.random.uniform((5, 4)))
print(f"Sequential count_params(): {seq_model.count_params()}")

# ---- 5. training=True/False ----
print("\n=== training=True/False ===")
dropout_demo = tf.keras.Sequential([tf.keras.layers.Dropout(0.9)])
same_input = tf.ones((1, 10))
train_out = dropout_demo(same_input, training=True)
eval_out = dropout_demo(same_input, training=False)
print(f"training=True output (mostly zeroed): {train_out.numpy()}")
print(f"training=False output (unchanged):    {eval_out.numpy()}")
