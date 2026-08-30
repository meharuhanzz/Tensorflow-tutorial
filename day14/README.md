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
mechanism `metrics=["accuracy"]` in `compile()` uses internally:

```python
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

for images, labels in val_ds:
    preds = tf.argmax(model(images, training=False), axis=1)
    precision.update_state(labels, preds)
    recall.update_state(labels, preds)

print(f"precision: {precision.result():.3f}, recall: {recall.result():.3f}")
```

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

## Inspecting individual predictions

Aggregate numbers tell you *that* something's wrong; looking at specific
predictions (especially wrong, confident ones) is often how you discover
*why* — an ambiguous image, a mislabeled example, or a genuinely hard
case worth more training data for.

## Try it yourself

1. Train Day 9's CNN on a 3-class shape dataset, collect
   `all_preds`/`all_labels` over the validation set, and print
   `tf.math.confusion_matrix(all_labels, all_preds)`.
2. Compute per-class accuracy from that confusion matrix and identify
   which class (if any) the model does worst on.
3. Compute precision and recall per class using `tf.keras.metrics.Precision`/
   `Recall` with a `class_id` argument (compute each metric once per
   class). Note these expect the *raw per-class scores* (logits/softmax
   output, shape `(num_samples, num_classes)`), not already-argmaxed
   predictions — pass `model(images, training=False)` directly, not
   `tf.argmax(...)` of it. Compare the results against what the
   confusion matrix already told you.
4. Pick the 5 validation examples the model got *most confidently wrong*
   (highest softmax probability on an incorrect class) and inspect them —
   is there a pattern (a specific shape/color combination, a boundary
   case)?
