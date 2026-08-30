# TensorFlow Day 15 — Capstone: A Full Image Classifier Project

Congratulations on making it through 14 days — this day combines
everything into one complete, real training project.

## What it does

Classifies 4 shapes (circle, square, triangle, star) from small synthetic
images, using a pretrained MobileNetV2 backbone, with a proper staged
fine-tuning schedule, early stopping, best-checkpoint saving, and a full
evaluation report at the end.

## Where each earlier day shows up

| Concept | Day | Where in this project |
|---|---|---|
| `tf.data.Dataset` | 6 | `train_ds` / `val_ds` |
| `image_dataset_from_directory` + preprocessing | 7 | loading the shapes, `preprocess_input` |
| CNN fundamentals | 8 | Understood, though today uses a pretrained one instead of training from scratch |
| A full train/val loop | 5, 9 | `model.fit(...)`, the training call |
| Overfitting & Dropout | 10 | `Dropout` in the new head, `EarlyStopping` callback |
| Transfer learning | 11 | `MobileNetV2(weights="imagenet")`, staged unfreezing |
| Saving/loading checkpoints | 12 | `ModelCheckpoint`, the reload-and-confirm section |
| GPU/device-agnostic code, mixed precision | 1, 13 | `tf.config.list_physical_devices`, `set_global_policy` |
| Evaluation & metrics | 14 | Confusion matrix, per-class accuracy, precision/recall |

## The staged fine-tuning schedule

```python
# Stage 1: head only, base frozen
base_model.trainable = False
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5)

# Stage 2: unfreeze the last block, much smaller LR
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss_fn, metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10,
          callbacks=[early_stop, checkpoint_cb])
```

Stage 1 trains only the new head (backbone frozen). Stage 2 unfreezes
MobileNetV2's last block too, at a smaller learning rate. This is the
same two-stage recipe from Day 11, now doing real work end to end.

## Early stopping + best-checkpoint saving, together

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras", monitor="val_loss", save_best_only=True)

model.fit(train_ds, validation_data=val_ds, epochs=30,
          callbacks=[early_stop, checkpoint_cb])
```

Both callbacks watch `val_loss`. `EarlyStopping(restore_best_weights=True)`
means that even if training runs a few extra epochs past the best one
before triggering, the in-memory `model` ends up holding the best
weights, not the last ones — and `ModelCheckpoint` independently writes
that same best point to disk as training goes, so you have it on disk
even if the process is interrupted before `fit()` returns.

## Confirming the saved model actually works

Don't just save a checkpoint and trust it — reload it into a fresh model
and confirm the predictions match, in the same script:

```python
reloaded = tf.keras.models.load_model("best_model.keras")
original_preds = model.predict(val_ds)
reloaded_preds = reloaded.predict(val_ds)
assert (original_preds == reloaded_preds).all()
```

This "does the save/load round-trip actually work" check is cheap
insurance worth having in any real project.

## Run it

```bash
python3 make_shapes.py   # once
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
