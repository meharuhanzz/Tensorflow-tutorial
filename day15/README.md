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

## Try it yourself

This capstone extends directly rather than starting fresh — exactly the
kind of work you'd do on a real project after getting an initial version
working:

1. Generate a 4-class shape dataset (circle/square/triangle/star) and
   load it with `image_dataset_from_directory` (Day 7/9).
2. Build the MobileNetV2-based model, run the two-stage fine-tuning
   schedule above, and get a full `tf.math.confusion_matrix` +
   per-class-accuracy report (Day 14) on the validation set.
3. Extend the two-stage schedule to three stages: head-only → unfreeze
   the last 20 layers → unfreeze the last 60 layers, shrinking the
   learning rate further at each stage. Does the extra stage help on
   this small a dataset, or is two stages already enough?
4. Add mixed precision (`set_global_policy("mixed_bfloat16")`, Day 13) to
   the whole pipeline and confirm the model still trains to a comparable
   final accuracy.

## Where to go from here

You've now seen the complete shape of a real TensorFlow/Keras project.
Natural next steps beyond this course: learning about more advanced
architectures (Vision Transformers via `tf.keras.applications` or
Hugging Face's `transformers` library, which has TensorFlow support
too), experiment tracking tools (TensorBoard — built into Keras via the
`TensorBoard` callback, worth a look even for small projects), and
larger, real-world datasets instead of synthetic shapes.
