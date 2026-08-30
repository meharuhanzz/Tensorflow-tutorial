"""TensorFlow Day 4 -- Loss Functions & Optimizers.

Run me with:  python3 main.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

# ---- 1. MeanSquaredError ----
print("=== MeanSquaredError ===")
mse = tf.keras.losses.MeanSquaredError()
targets = [1.0, 2.0, 3.0]
close_preds = [1.1, 2.2, 2.7]
far_preds = [5.0, -1.0, 8.0]
print(f"close predictions loss: {mse(targets, close_preds).numpy():.4f}")
print(f"far predictions loss:   {mse(targets, far_preds).numpy():.4f}")

# ---- 2. SparseCategoricalCrossentropy, from_logits ----
print("\n=== SparseCategoricalCrossentropy ===")
cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

true_label = tf.constant([0])
confident_correct = tf.constant([[5.0, 0.0, 0.0, 0.0]])
confident_wrong = tf.constant([[0.0, 0.0, 0.0, 5.0]])
unconfident = tf.constant([[0.1, 0.1, 0.1, 0.1]])

print(f"confident + correct loss:  {cce(true_label, confident_correct).numpy():.4f}")
print(f"unconfident loss:          {cce(true_label, unconfident).numpy():.4f}")
print(f"confident + WRONG loss:    {cce(true_label, confident_wrong).numpy():.4f}")

# ---- 3. The from_logits mistake ----
print("\n=== from_logits mistake ===")
cce_wrong_flag = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
print(f"same logits, from_logits=False (WRONG, don't do this): "
      f"{cce_wrong_flag(true_label, confident_correct).numpy():.4f}")
print("(this number is misleading -- the loss silently assumes the input")
print(" was already a probability distribution, which raw logits are not)")

# ---- 4. Optimizer: one manual training step ----
print("\n=== One manual training step with an optimizer ===")


class SimpleNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(8, activation="relu")
        self.layer2 = tf.keras.layers.Dense(1)

    def call(self, x):
        return self.layer2(self.layer1(x))


model = SimpleNet()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
loss_fn = tf.keras.losses.MeanSquaredError()

x = tf.random.uniform((8, 3))
y = tf.random.uniform((8, 1))

with tf.GradientTape() as tape:
    predictions = model(x, training=True)
    loss = loss_fn(y, predictions)
print(f"loss before step: {loss.numpy():.4f}")

grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

predictions_after = model(x, training=True)
loss_after = loss_fn(y, predictions_after)
print(f"loss after 1 step: {loss_after.numpy():.4f}")

# ---- 5. Learning the relationship y = 2x + 1, over several steps ----
print("\n=== Learning y = 2x + 1 from data ===")
x_line = tf.reshape(tf.linspace(-5.0, 5.0, 50), (-1, 1))
y_line = 2 * x_line + 1

linear_model = tf.keras.layers.Dense(1)
adam = tf.keras.optimizers.Adam(learning_rate=0.5)
mse_loss = tf.keras.losses.MeanSquaredError()

for step in range(60):
    with tf.GradientTape() as tape:
        preds = linear_model(x_line)
        loss = mse_loss(y_line, preds)
    grads = tape.gradient(loss, linear_model.trainable_variables)
    adam.apply_gradients(zip(grads, linear_model.trainable_variables))
    if step % 10 == 0 or step == 59:
        print(f"step {step}: loss = {loss.numpy():.4f}")

learned_w = linear_model.trainable_variables[0].numpy().item()
learned_b = linear_model.trainable_variables[1].numpy().item()
print(f"learned: y = {learned_w:.3f}x + {learned_b:.3f} (true: y = 2x + 1)")
