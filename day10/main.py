"""TensorFlow Day 10 -- Overfitting & Regularization.

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

# Deliberately tiny training set (3 images/class) to make overfitting show
# up clearly and quickly.
full_train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=1,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=16,
)
class_names = full_train_ds.class_names
num_classes = len(class_names)

rescale = tf.keras.layers.Rescaling(1.0 / 255)
# .repeat() matters here: with a single fixed batch and an explicit
# steps_per_epoch below, a NON-repeating dataset hits StopIteration at
# the epoch boundary on alternating epochs, silently producing a
# zeroed-out "phantom" epoch every other epoch (a real, reproducible
# Keras/tf.data quirk with tiny fixed-size datasets -- confirmed by
# running this without .repeat() and watching accuracy alternate with
# exact 0.0 epochs). .repeat() makes the dataset infinite so
# steps_per_epoch=1 cleanly slices off one batch per epoch instead.
tiny_train_ds = (full_train_ds.unbatch().take(9).cache()   # 3 images/class, 3 classes
                  .map(lambda x, y: (rescale(x), y)).batch(9).repeat())
val_ds = val_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def conv_block(filters):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
    ])


def build_cnn(dropout=0.0, l2=0.0):
    reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None
    layers = [conv_block(16), conv_block(32), tf.keras.layers.Flatten()]
    if dropout > 0:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg))
    layers.append(tf.keras.layers.Dense(num_classes))
    return tf.keras.Sequential(layers)


def train_and_report(name, model, epochs=40):
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    # tiny_train_ds is exactly one batch of 9 -- steps_per_epoch=1 makes
    # that explicit rather than relying on Keras's dataset-size guessing,
    # which otherwise intermittently cuts training short ("ran out of data").
    history = model.fit(tiny_train_ds, validation_data=val_ds, epochs=epochs,
                         steps_per_epoch=1, verbose=0, shuffle=False)
    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    print(f"{name:35s} train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  gap={train_acc - val_acc:.3f}")
    return history


print("=== Overfitting demo: 9-image training set, 40 epochs ===\n")
train_and_report("no regularization", build_cnn())
train_and_report("Dropout(0.5)", build_cnn(dropout=0.5))
train_and_report("L2(1e-2) on Dense", build_cnn(l2=1e-2))
train_and_report("Dropout(0.5) + L2(1e-2)", build_cnn(dropout=0.5, l2=1e-2))

# ---- EarlyStopping demo ----
print("\n=== EarlyStopping on a longer run ===")
model = build_cnn(dropout=0.5)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(tiny_train_ds, validation_data=val_ds, epochs=100,
                     steps_per_epoch=1, callbacks=[early_stop], verbose=0, shuffle=False)
epochs_run = len(history.history["loss"])
print(f"requested 100 epochs, EarlyStopping actually ran: {epochs_run}")
