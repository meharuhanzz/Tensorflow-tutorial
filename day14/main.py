"""TensorFlow Day 14 -- Evaluation & Metrics.

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

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=16,
)
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=16,
)
class_names = train_ds_raw.class_names
num_classes = len(class_names)
print(f"classes: {class_names}")

rescale = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds_raw.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds_raw.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)

# ---- Train a quick CNN so we have something real to evaluate ----
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes),
])
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=20, verbose=0, shuffle=False)

# ---- 1. Collecting predictions ----
print("\n=== Collecting predictions ===")
all_preds, all_labels, all_scores = [], [], []
for images, labels in val_ds:
    logits = model(images, training=False)
    preds = tf.argmax(logits, axis=1)
    all_preds.extend(preds.numpy().tolist())
    all_labels.extend(labels.numpy().tolist())
    all_scores.append(logits)
all_scores = tf.concat(all_scores, axis=0)
print(f"collected {len(all_preds)} predictions")

# ---- 2. Confusion matrix ----
print("\n=== Confusion matrix ===")
cm = tf.math.confusion_matrix(all_labels, all_preds, num_classes=num_classes)
print(f"rows = true class, columns = predicted class, order = {class_names}")
print(cm.numpy())

# ---- 3. Per-class accuracy ----
print("\n=== Per-class accuracy ===")
cmf = tf.cast(cm, tf.float32)
row_totals = tf.reduce_sum(cmf, axis=1)
per_class_acc = tf.linalg.diag_part(cmf) / tf.maximum(row_totals, 1.0)
for name, acc in zip(class_names, per_class_acc.numpy()):
    print(f"  {name:10s}: {acc:.3f}")

# ---- 4. Precision / recall per class ----
# class_id requires BOTH y_true and y_pred as one-hot/per-class arrays --
# sparse integer labels silently give wrong (near-zero) results here, an
# easy trap since SparseCategoricalCrossentropy (Day 4) takes sparse
# labels directly and this doesn't.
print("\n=== Precision / Recall per class ===")
all_labels_onehot = tf.one_hot(all_labels, num_classes)
all_probs = tf.nn.softmax(all_scores, axis=1)
for class_id, name in enumerate(class_names):
    precision = tf.keras.metrics.Precision(class_id=class_id)
    recall = tf.keras.metrics.Recall(class_id=class_id)
    precision.update_state(all_labels_onehot, all_probs)
    recall.update_state(all_labels_onehot, all_probs)
    print(f"  {name:10s}: precision={precision.result().numpy():.3f}, "
          f"recall={recall.result().numpy():.3f}")
print("(these use a 0.5 probability threshold per class, NOT argmax --")
print(" so they're in the same ballpark as the confusion-matrix numbers")
print(" above but won't match exactly; a class can lose the argmax to")
print(" another class while still being individually above 0.5, or vice versa)")

# ---- 5. Most confidently wrong predictions ----
print("\n=== Most confidently wrong predictions ===")
probs = tf.nn.softmax(all_scores, axis=1).numpy()
wrong_indices = [i for i in range(len(all_preds)) if all_preds[i] != all_labels[i]]
wrong_confidences = [(i, probs[i][all_preds[i]]) for i in wrong_indices]
wrong_confidences.sort(key=lambda t: -t[1])
for i, conf in wrong_confidences[:5]:
    print(f"  example {i}: true={class_names[all_labels[i]]}, "
          f"predicted={class_names[all_preds[i]]} (confidence {conf:.3f})")
if not wrong_confidences:
    print("  (no wrong predictions on this validation set -- try fewer epochs")
    print("   or a smaller model if you want to see some)")
