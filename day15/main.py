"""TensorFlow Day 15 -- Capstone: A Full Image Classifier Project.

Run `python3 make_shapes.py` once first, then:  python3 main.py

Classifies 4 shapes (circle, square, triangle, star) using a pretrained
MobileNetV2 backbone, staged fine-tuning, early stopping, best-checkpoint
saving, and a full evaluation report -- everything from Days 1-14.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "shapes")
OUT_DIR = os.path.join(HERE, "_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.isdir(DATA_DIR):
    raise SystemExit("Run `python3 make_shapes.py` first to generate shapes/.")

IMG_SIZE = (96, 96)
BATCH_SIZE = 16

# ---- Day 1/13: device check ----
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs visible: {len(gpus)}")

# ---- Day 6/7/9: tf.data.Dataset from folders ----
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
class_names = train_ds_raw.class_names
num_classes = len(class_names)
print(f"classes: {class_names}")

train_ds = train_ds_raw.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds_raw.prefetch(tf.data.AUTOTUNE)

# ---- Day 11: transfer learning -- MobileNetV2 backbone + new head ----
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)   # Day 7
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)                    # Day 8
x = tf.keras.layers.Dropout(0.3)(x)                                  # Day 10
outputs = tf.keras.layers.Dense(num_classes)(x)
model = tf.keras.Model(inputs, outputs)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ---- Day 10/12: callbacks -- early stopping + best-checkpoint saving ----
best_path = os.path.join(OUT_DIR, "best_model.keras")
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    best_path, monitor="val_loss", save_best_only=True)

# ---- Stage 1: head-only ----
print("\n=== Stage 1: head-only (up to 10 epochs, early stopping active) ===")
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
history1 = model.fit(train_ds, validation_data=val_ds, epochs=10,
                      callbacks=[early_stop, checkpoint_cb], verbose=0, shuffle=False)
print(f"stage 1: ran {len(history1.history['loss'])} epochs, "
      f"final val_acc = {history1.history['val_accuracy'][-1]:.3f}")

# ---- Stage 2: unfreeze last 20 layers, smaller LR ----
print("\n=== Stage 2: unfreeze last 20 layers (up to 10 epochs) ===")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

early_stop2 = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint_cb2 = tf.keras.callbacks.ModelCheckpoint(
    best_path, monitor="val_loss", save_best_only=True)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss_fn, metrics=["accuracy"])
history2 = model.fit(train_ds, validation_data=val_ds, epochs=10,
                      callbacks=[early_stop2, checkpoint_cb2], verbose=0, shuffle=False)
print(f"stage 2: ran {len(history2.history['loss'])} epochs, "
      f"final val_acc = {history2.history['val_accuracy'][-1]:.3f}")

# ---- Day 12: confirm the saved checkpoint actually round-trips ----
print("\n=== Verifying the save/load round-trip ===")
reloaded = tf.keras.models.load_model(best_path)
one_batch = next(iter(val_ds))[0]
original_preds = model(one_batch, training=False)
reloaded_preds = reloaded(one_batch, training=False)
matches = bool(tf.reduce_all(tf.abs(original_preds - reloaded_preds) < 1e-5))
print(f"reloaded model's predictions match the in-memory model: {matches}")

# ---- Day 14: full evaluation report ----
print("\n=== Evaluation report ===")
all_preds, all_labels = [], []
for images, labels in val_ds:
    logits = model(images, training=False)
    preds = tf.argmax(logits, axis=1)
    all_preds.extend(preds.numpy().tolist())
    all_labels.extend(labels.numpy().tolist())

cm = tf.math.confusion_matrix(all_labels, all_preds, num_classes=num_classes)
print(f"confusion matrix (rows=true, cols=pred, order={class_names}):")
print(cm.numpy())

cmf = tf.cast(cm, tf.float32)
per_class_acc = tf.linalg.diag_part(cmf) / tf.maximum(tf.reduce_sum(cmf, axis=1), 1.0)
for name, acc in zip(class_names, per_class_acc.numpy()):
    print(f"  {name:10s} accuracy: {acc:.3f}")

overall_acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
print(f"\noverall validation accuracy: {overall_acc:.3f}")
