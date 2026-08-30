"""TensorFlow Day 5 -- The Training Loop.

Run me with:  python3 main.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

# ---- Generate three synthetic 2D clusters ----
print("=== Generating synthetic data ===")
tf.random.set_seed(0)
n_per_class = 60
centers = [(-3.0, -3.0), (3.0, -3.0), (0.0, 3.0)]
X_parts, Y_parts = [], []
for label, (cx, cy) in enumerate(centers):
    pts = tf.random.normal((n_per_class, 2)) + tf.constant([cx, cy])
    X_parts.append(pts)
    Y_parts.append(tf.fill((n_per_class,), label))
X = tf.concat(X_parts, axis=0)
Y = tf.concat(Y_parts, axis=0)
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# ---- Train/test split ----
indices = tf.random.shuffle(tf.range(len(X)))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]
X_train, Y_train = tf.gather(X, train_idx), tf.gather(Y, train_idx)
X_test, Y_test = tf.gather(X, test_idx), tf.gather(Y, test_idx)
print(f"train: {len(X_train)} examples, test: {len(X_test)} examples")


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(3),   # 3 logits, no activation
    ])


def accuracy(logits, labels):
    preds = tf.argmax(logits, axis=1, output_type=labels.dtype)
    return tf.reduce_mean(tf.cast(preds == labels, tf.float32))


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ---- Way 1: the custom loop ----
print("\n=== Way 1: custom GradientTape loop ===")
model_custom = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

for epoch in range(100):
    with tf.GradientTape() as tape:
        logits = model_custom(X_train, training=True)
        loss = loss_fn(Y_train, logits)
    grads = tape.gradient(loss, model_custom.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_custom.trainable_variables))
    if epoch % 20 == 0 or epoch == 99:
        train_acc = accuracy(logits, Y_train)
        print(f"epoch {epoch}: loss = {loss.numpy():.4f}, train_acc = {train_acc.numpy():.3f}")

test_logits = model_custom(X_test, training=False)   # Day 2: no tape at all
test_acc_custom = accuracy(test_logits, Y_test)
print(f"custom loop -- final test accuracy: {test_acc_custom.numpy():.3f}")

# ---- Way 2: model.fit() ----
print("\n=== Way 2: model.fit() ===")
model_fit = build_model()
model_fit.compile(optimizer=tf.keras.optimizers.Adam(0.05),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=["accuracy"])
history = model_fit.fit(X_train, Y_train, epochs=100,
                         validation_data=(X_test, Y_test), verbose=0)
print(f"fit() -- final train acc: {history.history['accuracy'][-1]:.3f}, "
      f"final val acc: {history.history['val_accuracy'][-1]:.3f}")

test_loss, test_acc_fit = model_fit.evaluate(X_test, Y_test, verbose=0)
print(f"fit() -- model.evaluate() test accuracy: {test_acc_fit:.3f}")

print(f"\nboth approaches roughly agree: "
      f"custom={test_acc_custom.numpy():.3f} vs fit={test_acc_fit:.3f}")
