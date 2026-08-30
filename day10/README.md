# TensorFlow Day 10 — Overfitting & Regularization

## What overfitting looks like

A model **overfits** when it starts memorizing the specific training
examples instead of learning the general pattern behind them. The
telltale sign: **training accuracy keeps climbing while validation
accuracy plateaus or gets worse** — a growing gap between the two. Train
an unregularized CNN (Day 8/9) on a tiny training set (a couple dozen
images) for 40 epochs and watch `history.history["accuracy"]` pull away
from `history.history["val_accuracy"]` — that gap is overfitting, made
visible.

## `tf.keras.layers.Dropout` — random redundancy

```python
tf.keras.layers.Dropout(rate=0.5)
```

During **training**, Dropout randomly zeroes out each activation with
probability `rate`. This forces the network not to over-rely on any
single neuron — since that neuron might be "switched off" on any given
step, the network is pushed toward more redundant, robust
representations.

**Critically, Dropout only does this during training.** During inference
(`training=False`), it does nothing at all — every activation passes
through unchanged. This is exactly why the `training=True/False`
distinction from Day 3 matters in practice: Dropout literally checks that
argument to decide how to behave (`model.fit()`/`.evaluate()` set it for
you automatically).

## L2 weight regularization — penalizing large weights

```python
tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-3))
```

Adding a `kernel_regularizer` to a layer nudges its weights slightly
toward zero on each update step, added directly into the loss the
optimizer is minimizing (this is the "weight decay" idea from PyTorch,
but implemented as a loss-side penalty rather than an optimizer
argument — functionally similar in effect, worth knowing the mechanism
differs). This discourages the model from relying on any single weight
growing extremely large to fit the training data exactly — another form
of "don't over-commit to what you've memorized."

## `EarlyStopping` — a Keras callback

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[early_stop])
```

Rather than training a fixed number of epochs regardless of what's
happening, `EarlyStopping` watches `val_loss` and stops training once it
hasn't improved for `patience` epochs — `restore_best_weights=True`
rolls the model back to its best checkpoint rather than leaving it at
whatever it degraded to afterward. This is a `callbacks=[...]` argument
to `model.fit()` — Keras's general mechanism for hooking extra behavior
into the training loop without writing a custom one (you'll use
callbacks again on Day 12).

## Other regularization you already have

- **Data augmentation** (Day 7) — effectively gives the model more,
  varied training examples to learn from, rather than the same fixed set
  repeated every epoch.
- **A bigger training set** — often the single most effective fix, when
  you're able to get one.

## The tradeoff

Regularization usually makes training accuracy climb a little slower or
plateau a little lower — that's expected, and usually worth it, because
what you actually care about is performance on data the model **hasn't**
seen, which is what validation accuracy measures.

## Run it

```bash
python3 make_shapes.py   # once
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
