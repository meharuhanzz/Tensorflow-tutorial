"""TensorFlow Day 11 -- Transfer Learning.

Run `python3 make_shapes.py` once first, then:  python3 main.py

Note: images get resized to 96x96, so this runs a bit slower than
earlier days' 32x32/48x48 examples -- still fine on CPU for this small a
dataset (MobileNetV2 requires input at least 96x96).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "shapes")

if not os.path.isdir(DATA_DIR):
    raise SystemExit("Run `python3 make_shapes.py` first to generate shapes/.")

IMG_SIZE = (96, 96)
BATCH_SIZE = 16

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
class_names = train_ds_raw.class_names   # grab this BEFORE .prefetch() wraps the dataset
num_classes = len(class_names)
print(f"classes: {class_names}")

train_ds = train_ds_raw.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds_raw.prefetch(tf.data.AUTOTUNE)

# ---- 1. Loading the pretrained model ----
print("\n=== Loading MobileNetV2 (ImageNet weights) ===")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet",
)
print(f"base model layers: {len(base_model.layers)}")

# ---- 2. Build the head ----
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(num_classes)(x)
model = tf.keras.Model(inputs, outputs)

total_params = model.count_params()
trainable_params = sum(int(tf.size(v)) for v in model.trainable_variables)
print(f"total params: {total_params}, trainable (head only): {trainable_params}")

# ---- 3. Stage 1: train the head only ----
print("\n=== Stage 1: head-only training (5 epochs) ===")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
history1 = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0, shuffle=False)
print(f"after stage 1: train_acc={history1.history['accuracy'][-1]:.3f}, "
      f"val_acc={history1.history['val_accuracy'][-1]:.3f}")

# ---- 4. Stage 2: unfreeze the last 20 layers, smaller LR ----
print("\n=== Stage 2: unfreeze last 20 layers, fine-tune (5 epochs) ===")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

trainable_params_stage2 = sum(int(tf.size(v)) for v in model.trainable_variables)
print(f"trainable params now: {trainable_params_stage2} (was {trainable_params})")

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss_fn, metrics=["accuracy"])
history2 = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0, shuffle=False)
print(f"after stage 2: train_acc={history2.history['accuracy'][-1]:.3f}, "
      f"val_acc={history2.history['val_accuracy'][-1]:.3f}")

print(f"\nsummary: head-only val_acc={history1.history['val_accuracy'][-1]:.3f} "
      f"-> fine-tuned val_acc={history2.history['val_accuracy'][-1]:.3f}")
