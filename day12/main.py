"""TensorFlow Day 12 -- Saving, Loading & Checkpointing.

Run me with:  python3 main.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1),
    ])


x_sample = tf.random.uniform((4, 3))

# ---- 1. Saving and loading just the weights ----
print("=== save_weights / load_weights ===")
model = build_model()
model(x_sample)   # build it once
weights_path = os.path.join(OUT_DIR, "weights.weights.h5")
model.save_weights(weights_path)

new_model = build_model()
new_model(x_sample)   # must be built (called once) before weights can be loaded in
new_model.load_weights(weights_path)

original_out = model(x_sample)
loaded_out = new_model(x_sample)
print(f"predictions match after reload: {bool(tf.reduce_all(original_out == loaded_out))}")

# ---- 2. Saving the whole model ----
print("\n=== model.save() / load_model() -- architecture + weights ===")
full_path = os.path.join(OUT_DIR, "full_model.keras")
model.save(full_path)
loaded_model = tf.keras.models.load_model(full_path)
print(f"reloaded model output matches: "
      f"{bool(tf.reduce_all(model(x_sample) == loaded_model(x_sample)))}")

# ---- 3. tf.train.Checkpoint -- full state, including the optimizer ----
print("\n=== tf.train.Checkpoint -- resuming training exactly ===")
ckpt_model = build_model()
ckpt_model(x_sample)
optimizer = tf.keras.optimizers.Adam(0.01)
loss_fn = tf.keras.losses.MeanSquaredError()

y_sample = tf.random.uniform((4, 1))
for step in range(5):
    with tf.GradientTape() as tape:
        loss = loss_fn(y_sample, ckpt_model(x_sample, training=True))
    grads = tape.gradient(loss, ckpt_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, ckpt_model.trainable_variables))
print(f"loss after 5 steps (before checkpointing): {loss.numpy():.4f}")

checkpoint = tf.train.Checkpoint(model=ckpt_model, optimizer=optimizer)
ckpt_dir = os.path.join(OUT_DIR, "ckpt_dir")
saved_path = checkpoint.save(os.path.join(ckpt_dir, "ckpt"))
print(f"checkpoint saved at: {saved_path}")

# restore into a FRESH model + optimizer pair
fresh_model = build_model()
fresh_model(x_sample)
fresh_optimizer = tf.keras.optimizers.Adam(0.01)
fresh_checkpoint = tf.train.Checkpoint(model=fresh_model, optimizer=fresh_optimizer)
fresh_checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

# one more training step on the RESTORED pair -- should continue smoothly,
# not spike (a spike would mean optimizer state wasn't really restored)
with tf.GradientTape() as tape:
    restored_loss = loss_fn(y_sample, fresh_model(x_sample, training=True))
grads = tape.gradient(restored_loss, fresh_model.trainable_variables)
fresh_optimizer.apply_gradients(zip(grads, fresh_model.trainable_variables))
print(f"loss right after restoring + 1 more step: {restored_loss.numpy():.4f} "
      f"(close to the pre-save loss, not a spike)")

# ---- 4. ModelCheckpoint callback -- save-best-only ----
# A small training set + an oversized model, trained for many epochs, so
# it genuinely overfits and val_loss gets WORSE well before the final
# epoch -- otherwise "best" and "last" checkpoint would trivially be the
# same thing and this demo wouldn't show anything.
print("\n=== ModelCheckpoint(save_best_only=True) ===")
tf.random.set_seed(1)
X = tf.random.normal((60, 3))
Y = tf.cast(tf.reduce_sum(X, axis=1) > 0, tf.int32)
X_train, Y_train = X[:12], Y[:12]
X_val, Y_val = X[12:], Y[12:]

cb_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(2),
])
cb_model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

best_path = os.path.join(OUT_DIR, "best_model.keras")
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    best_path, monitor="val_loss", save_best_only=True)

history = cb_model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                        epochs=60, callbacks=[checkpoint_cb], verbose=0)
val_losses = history.history["val_loss"]
best_epoch = int(tf.argmin(val_losses))
last_epoch = len(val_losses) - 1
print(f"val_loss: epoch {best_epoch} = {val_losses[best_epoch]:.4f} (best) vs. "
      f"epoch {last_epoch} = {val_losses[last_epoch]:.4f} (last, overfit)")
print(f"{best_path} holds the epoch-{best_epoch} weights, not the worse final-epoch ones.")
