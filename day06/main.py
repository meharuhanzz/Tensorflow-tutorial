"""TensorFlow Day 6 -- tf.data.Dataset.

Run me with:  python3 main.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

tf.random.set_seed(0)

# ---- 1. Building a Dataset from in-memory arrays ----
print("=== from_tensor_slices, no batching ===")
X = tf.random.normal((100, 2))
Y = tf.random.uniform((100,), maxval=2, dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
for i, (x, y) in enumerate(dataset.take(3)):
    print(f"example {i}: x={x.numpy()}, y={y.numpy()}")

# ---- 2. .shuffle().batch() -- chained ----
print("\n=== shuffle + batch ===")
batched = dataset.shuffle(buffer_size=100).batch(16)
sizes = [b[0].shape[0] for b in batched]
print(f"batch sizes: {sizes} (100 isn't evenly divisible by 16 -- last batch is smaller)")

# ---- 3. Order matters: shuffle before batch ----
print("\n=== shuffle AFTER batch (usually wrong) ===")
wrong_order = dataset.batch(16).shuffle(buffer_size=7)
wrong_sizes = [b[0].shape[0] for b in wrong_order]
print(f"batch sizes: {wrong_sizes} -- still valid batches, but now whole BATCHES")
print("are shuffled relative to each other, not the individual examples inside them.")

# ---- 4. .prefetch() ----
print("\n=== prefetch (no visible output -- a performance detail) ===")
pipeline = dataset.shuffle(100).batch(16).prefetch(tf.data.AUTOTUNE)
print("pipeline built with .prefetch(tf.data.AUTOTUNE) -- doesn't change results,")
print("only lets the next batch prepare while the current one trains.")

# ---- 5. Reusing Day 5's classifier, now with a real Dataset pipeline ----
print("\n=== Day 5's classifier, trained via tf.data.Dataset ===")
n_per_class = 60
centers = [(-3.0, -3.0), (3.0, -3.0), (0.0, 3.0)]
X_parts, Y_parts = [], []
for label, (cx, cy) in enumerate(centers):
    pts = tf.random.normal((n_per_class, 2)) + tf.constant([cx, cy])
    X_parts.append(pts)
    Y_parts.append(tf.fill((n_per_class,), label))
X_cluster = tf.concat(X_parts, axis=0)
Y_cluster = tf.concat(Y_parts, axis=0)

train_ds = (tf.data.Dataset.from_tensor_slices((X_cluster, Y_cluster))
            .shuffle(len(X_cluster))
            .batch(16)
            .prefetch(tf.data.AUTOTUNE))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3),
])
optimizer = tf.keras.optimizers.Adam(0.05)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for epoch in range(20):
    epoch_loss = 0.0
    n_batches = 0
    for batch_x, batch_y in train_ds:
        with tf.GradientTape() as tape:
            logits = model(batch_x, training=True)
            loss = loss_fn(batch_y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += loss.numpy()
        n_batches += 1
    if epoch % 5 == 0 or epoch == 19:
        print(f"epoch {epoch}: avg batch loss = {epoch_loss / n_batches:.4f}")
