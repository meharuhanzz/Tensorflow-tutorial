# TensorFlow Day 4 — Loss Functions & Optimizers

Two pieces glue `GradientTape` (Day 2) and models (Day 3) into something
that actually learns: a **loss function** that measures how wrong the
model is, and an **optimizer** that uses the resulting gradients to
improve it.

## Loss functions

**`tf.keras.losses.MeanSquaredError`** — for regression, predicting a
continuous number:

```python
mse = tf.keras.losses.MeanSquaredError()
loss = mse(targets, predictions)   # mean of (target - prediction)^2
```

Note the argument order: Keras losses take `(y_true, y_pred)` —
targets first, predictions second. This is the opposite order from some
other frameworks and a genuinely easy mistake to make.

**`tf.keras.losses.SparseCategoricalCrossentropy`** — for classification,
predicting which of N classes something belongs to, when your labels are
plain integers (`0, 1, 2, ...`) rather than one-hot vectors. This is the
loss used throughout the rest of this course's classification days.

```python
cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = cce(true_labels, logits)
```

Important detail: `from_logits=True` tells Keras your model's output is
**raw scores (logits)**, not already-softmaxed probabilities — it applies
softmax internally, more numerically stably than doing it yourself.
Leaving `from_logits` at its default (`False`) while feeding in raw
logits gives silently wrong results — one of the most common Keras
mistakes, worth double-checking any time loss numbers look strange.

(If your labels are already one-hot vectors instead of integers, the
non-"Sparse" `CategoricalCrossentropy` is the equivalent.)

Confident + correct predictions produce low loss; confident + wrong
predictions produce high loss — try both in the exercises below and
compare the numbers.

## Optimizers

On Day 2 you updated a single weight by hand:

```python
w.assign_sub(learning_rate * grad)
```

An **optimizer** does exactly this, automatically, for *every* trainable
variable in your model:

```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

with tf.GradientTape() as tape:
    predictions = model(x, training=True)
    loss = loss_fn(y, predictions)
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

This four-line sequence — tape the forward pass, get the gradients, zip
them with the variables, `apply_gradients` — is the heart of every custom
TensorFlow training loop you'll ever write. Day 5 wraps it in an actual
loop over multiple epochs. (Notice there's no `zero_grad()` step here —
that's Day 2's tape-is-single-use behavior making PyTorch's "gradients
silently accumulate" gotcha simply not exist in TensorFlow.)

## SGD vs Adam

- **SGD** — the classic, simple algorithm: `param -= lr * grad`
  (optionally smoothed with momentum via `tf.keras.optimizers.SGD(momentum=0.9)`).
- **Adam** — adapts the effective learning rate per-parameter
  automatically. Usually converges faster with less manual tuning — the
  default choice for most modern deep learning work, including the
  transfer-learning day later in this course.

Switching between them is usually just changing one line:

```python
tf.keras.optimizers.SGD(learning_rate=0.01)
tf.keras.optimizers.Adam(learning_rate=0.01)
```

## Try it yourself

1. Compute `MeanSquaredError()(targets, predictions)` for
   `targets = [1.0, 2.0, 3.0]` and two different `predictions` arrays —
   one close to `targets`, one far off. Confirm the loss is bigger for
   the worse predictions.
2. Build 3 fake "logits" for a 4-class problem — one that's confidently
   correct, one that's confidently *wrong*, one that's unconfident/flat —
   and compute `SparseCategoricalCrossentropy(from_logits=True)` for each
   against the same true label. Order them from lowest to highest loss
   and confirm it matches your intuition.
3. Repeat exercise 2 but set `from_logits=False` by mistake (feeding the
   same raw logits in) — observe how the loss values change and why
   that's a bug, not a feature.
4. Take Day 3's `SimpleNet`, run one manual training step by hand (tape,
   gradients, `apply_gradients`) against a single batch of random data
   and a `MeanSquaredError` loss against random targets. Print the loss
   before and after the step and confirm it went down.
