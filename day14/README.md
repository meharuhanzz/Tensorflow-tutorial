# TensorFlow Day 14 — Evaluation & Metrics

A single accuracy number can hide a lot. Today: the tools for actually
understanding *how* a model is right or wrong, not just *how often*.

## Collecting predictions first

Metrics beyond a running accuracy average need every prediction and every
true label gathered together:

```python
all_preds, all_labels = [], []
for images, labels in val_ds:
    logits = model(images, training=False)   # Day 2/3 -- no tape, inference mode
    preds = tf.argmax(logits, axis=1)
    all_preds.extend(preds.numpy().tolist())
    all_labels.extend(labels.numpy().tolist())
```

## Confusion matrix — which classes get confused with which

TensorFlow has its own, so you don't need `sklearn` for this one:

```python
cm = tf.math.confusion_matrix(all_labels, all_preds)
```

Each row is a true class, each column a predicted class. The diagonal is
correct predictions; everything off-diagonal is a specific *kind* of
mistake — `cm[0][1]` tells you exactly how many true class-0 examples
got wrongly predicted as class 1.

## Per-class accuracy

```python
cm = tf.cast(cm, tf.float32)
per_class_acc = tf.linalg.diag_part(cm) / tf.reduce_sum(cm, axis=1)
```

The diagonal divided by each row's total. An overall accuracy of 98% can
still hide one specific class sitting at 40%, and per-class accuracy is
how you catch that.

## `tf.keras.metrics` — trackable, stateful metrics

Beyond one-shot computations like the confusion matrix above, Keras has
built-in metric *objects* that accumulate across batches — the same
mechanism `metrics=["accuracy"]` in `compile()` uses internally. For
**multi-class** precision/recall, use `class_id` to ask "how good is the
model specifically at class X" — one metric object per class:

```python
labels_onehot = tf.one_hot(all_labels, num_classes)      # NOT sparse integers
probs = tf.nn.softmax(model(images, training=False))       # NOT argmax predictions

precision = tf.keras.metrics.Precision(class_id=0)
recall = tf.keras.metrics.Recall(class_id=0)
precision.update_state(labels_onehot, probs)
recall.update_state(labels_onehot, probs)
print(f"class 0 -- precision: {precision.result():.3f}, recall: {recall.result():.3f}")
```

Two real gotchas here, both easy to get wrong the first time:

1. **`Precision()`/`Recall()` without `class_id` are binary metrics.**
   Feeding them plain multi-class integer labels (`0`, `1`, `2`, ...)
   directly produces numbers that *look* plausible but are meaningless —
   they're built for a two-class 0/1 world, not N classes.
2. **With `class_id`, both `y_true` *and* `y_pred` must be one-hot /
   per-class-score arrays**, shape `(num_samples, num_classes)` — sparse
   integer labels (the format `SparseCategoricalCrossentropy` on Day 4
   wants) silently give wrong results here instead of an error.

- **Precision** — of everything predicted as class X, how much of it
  really was X? (Are we crying wolf too often?)
- **Recall** — of everything that really was class X, how much did we
  actually catch? (Are we missing too many?)
- **F1** is the harmonic mean of the two — `2 * (p * r) / (p + r)`, or
  use `tf.keras.metrics.F1Score` directly.

These matter most when classes are imbalanced or when false positives
and false negatives have different real-world costs — plain accuracy
alone can look fine while one of these is quietly bad. `.reset_state()`
clears a metric object's accumulated state, e.g. between epochs.

**One more subtlety**: `Precision`/`Recall` default to a `0.5`
probability threshold per class, not "whichever class scored highest"
(argmax). So these numbers will be in the same ballpark as — but won't
exactly match — the per-class accuracy computed from the confusion
matrix above, which *is* argmax-based. Both are valid, answering
slightly different questions.

## Inspecting individual predictions

Aggregate numbers tell you *that* something's wrong; looking at specific
predictions (especially wrong, confident ones) is often how you discover
*why* — an ambiguous image, a mislabeled example, or a genuinely hard
case worth more training data for.

## Run it

```bash
python3 make_shapes.py   # once
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
