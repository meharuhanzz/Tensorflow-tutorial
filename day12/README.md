# TensorFlow Day 12 — Saving, Loading & Checkpointing

## Saving and loading just the weights

```python
model.save_weights("weights.weights.h5")

new_model = SmallNet()                 # recreate the architecture from your code
new_model(dummy_input)                   # build it once (Day 3's lazy-build rule) if it's a subclassed Model
new_model.load_weights("weights.weights.h5")
```

Notice: the architecture itself isn't saved — only the numbers. Your
Python class/`Sequential` definition is what recreates the structure;
loading just fills in the learned values. This means the model you load
into must match the one that produced the saved weights (same layer
names, same shapes) — and for a subclassed `Model`, it must have already
been called once on some input so its weights actually exist to be
overwritten (Day 3's "unbuilt" trap again).

## Saving the whole model — architecture and all

```python
model.save("full_model.keras")

loaded_model = tf.keras.models.load_model("full_model.keras")
```

Unlike PyTorch (where saving the whole object via `torch.save(model, path)`
is a fragile pickle of the actual Python object, discouraged), Keras's
`.keras` format saves architecture, weights, and the training
configuration (optimizer, loss, metrics from `compile()`) in one portable
file that reliably reconstructs the model without you needing the
original class definition available. This is Keras's recommended
approach for most use cases — a real, and pleasant, difference from
PyTorch's convention here.

## Full checkpoints — for resuming training exactly

The `.keras` format above is great for *using* a trained model, but
resuming interrupted *training* also needs the optimizer's internal state
(Adam tracks per-parameter moving averages, for example). `tf.train.Checkpoint`
handles this:

```python
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=tf.Variable(0))
checkpoint.save("ckpt_dir/ckpt")     # writes ckpt_dir/ckpt-1, -2, ... each call

# later, to resume:
checkpoint.restore(tf.train.latest_checkpoint("ckpt_dir"))
```

Loading it back restores all three pieces, so training can continue
exactly where it left off, rather than the optimizer having to "warm up"
its internal state again from scratch.

## `ModelCheckpoint` — save-best-only, as a callback

The manual "only save if improved" pattern from other frameworks is a
one-line callback in Keras:

```python
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",
    save_best_only=True,   # only overwrite when val_loss improves
)
model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[checkpoint_cb])
```

This guarantees the saved file is always the *best-performing* checkpoint
seen so far — not just whatever the training loop happened to produce on
its last epoch, which (especially without early stopping, Day 10's
lesson — and note `EarlyStopping` and `ModelCheckpoint` are commonly
passed together in the same `callbacks=[...]` list) might actually be a
worse, overfit version.

## Run it

```bash
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
