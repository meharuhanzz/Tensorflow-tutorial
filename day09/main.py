"""TensorFlow Day 9 -- Training a CNN End-to-End.

Run `python3 make_shapes.py` once first, then:  python3 main.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "shapes")

if not os.path.isdir(DATA_DIR):
    raise SystemExit("Run `python3 make_shapes.py` first to generate shapes/.")

IMG_SIZE = (48, 48)
BATCH_SIZE = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
class_names = train_ds.class_names
print(f"classes: {class_names}")

rescale = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)


def build_cnn(num_classes):
    def conv_block(filters):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
        ])
    return tf.keras.Sequential([
        conv_block(16),
        conv_block(32),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes),
    ])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ---- Way 1: model.fit() ----
print("\n=== model.fit() ===")
model_fit = build_cnn(len(class_names))
model_fit.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
history = model_fit.fit(train_ds, validation_data=val_ds, epochs=15, verbose=0, shuffle=False)
for e in (0, 4, 9, 14):
    print(f"epoch {e}: acc={history.history['accuracy'][e]:.3f}, "
          f"val_acc={history.history['val_accuracy'][e]:.3f}")

# ---- Way 2: custom loop with train_one_epoch()/evaluate() ----
print("\n=== Custom loop ===")
model_custom = build_cnn(len(class_names))
optimizer = tf.keras.optimizers.Adam()


def train_one_epoch(ds):
    for x, y in ds:
        with tf.GradientTape() as tape:
            logits = model_custom(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model_custom.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_custom.trainable_variables))


def evaluate(ds):
    correct, total = 0, 0
    for x, y in ds:
        logits = model_custom(x, training=False)   # no tape at all -- Day 2
        preds = tf.argmax(logits, axis=1)
        correct += int(tf.reduce_sum(tf.cast(preds == tf.cast(y, preds.dtype), tf.int32)))
        total += len(y)
    return correct / total


for epoch in range(15):
    train_one_epoch(train_ds)
    if epoch % 5 == 0 or epoch == 14:
        val_acc = evaluate(val_ds)
        print(f"epoch {epoch}: val_acc = {val_acc:.3f}")

print(f"\nfinal fit() val_acc:    {history.history['val_accuracy'][-1]:.3f}")
print(f"final custom val_acc:   {evaluate(val_ds):.3f}")
