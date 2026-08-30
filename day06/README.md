# TensorFlow Day 6 — `tf.data.Dataset`

Day 5 passed an entire small dataset through the model in one shot (or
let `model.fit()` handle batching invisibly). Real datasets can have
millions of examples — far too much to fit in memory or on a GPU at
once. The fix is **mini-batching**: process the data in small chunks,
updating weights after each one. `tf.data.Dataset` is TensorFlow's
pipeline for this.

## Building a Dataset from arrays already in memory

```python
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
```

This is the TensorFlow equivalent of writing a custom `__len__`/`__getitem__`
class in PyTorch — `from_tensor_slices` slices your arrays along their
first dimension automatically, so `dataset` now yields `(X[0], Y[0])`,
`(X[1], Y[1])`, ... one example at a time. You rarely need to write a
class by hand for in-memory data the way PyTorch requires.

## `.shuffle()`, `.batch()` — chained, in order

```python
dataset = (tf.data.Dataset.from_tensor_slices((X, Y))
           .shuffle(buffer_size=len(X))
           .batch(16))

for batch_x, batch_y in dataset:
    ...
```

`tf.data.Dataset` methods are designed to be **chained** — each one
returns a new dataset with that transformation applied, so you build a
pipeline by stacking `.method()` calls. `.shuffle(buffer_size)` fills a
buffer of that many examples and samples randomly from it (for a dataset
that fits in memory, set `buffer_size` to the full dataset length for a
true shuffle); `.batch(16)` groups consecutive examples into batches of
16. **Order matters**: shuffle *before* you batch, or you'll shuffle the
batches instead of the examples inside them.

## The Day 5 loop, now with real batching

```python
for epoch in range(num_epochs):
    for batch_x, batch_y in dataset:   # one batch at a time, not everything at once
        with tf.GradientTape() as tape:
            logits = model(batch_x, training=True)
            loss = loss_fn(batch_y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

Same tape-plus-`apply_gradients` core from Day 5 — it's now just nested
inside a loop over batches, which is nested inside a loop over epochs.
This nested-loop shape (epochs → batches → the training-step lines) is
what basically every real custom TensorFlow training script looks like.
(`model.fit(X, Y, batch_size=16)` does this same batching internally if
you're using the Keras loop instead.)

## `.shuffle()` on train data, skip it on validation/test

Shuffle **training** data every epoch — otherwise the model always sees
examples in the same order and can pick up on that ordering as a
spurious pattern, rather than the real relationship between input and
label. Skip `.shuffle()` on **validation/test** datasets — order doesn't
matter for evaluation, and keeping it fixed makes results easier to
inspect and compare run to run.

## `.prefetch()` — a performance detail worth knowing early

```python
dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)
```

`.prefetch()` lets TensorFlow prepare the *next* batch on the CPU while
the *current* batch is still training on the GPU, instead of the two
happening strictly one after another. `tf.data.AUTOTUNE` lets TensorFlow
pick how much to prefetch automatically. This doesn't change what your
model learns at all — only how fast training runs — but it's cheap
("free" one extra line) and standard enough on every real pipeline that
it's worth forming the habit now.

## Run it

```bash
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
