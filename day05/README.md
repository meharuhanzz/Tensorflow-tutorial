# TensorFlow Day 5 — The Training Loop

Everything from Days 1-4 comes together today into the complete pattern
you'll reuse for every model you ever train — written two ways, because
both are genuinely useful to know in TensorFlow.

## Way 1: the custom loop (mirrors Day 2/4 exactly)

```python
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        logits = model(X_train, training=True)   # forward pass
        loss = loss_fn(Y_train, logits)             # how wrong are we?

    grads = tape.gradient(loss, model.trainable_variables)   # Day 2
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # Day 4
```

That's it — the tape-plus-`apply_gradients` block is the entire
"learning" mechanism, in a loop. Everything else in deep learning
(bigger models, more data, fancier architectures) builds on top of this
exact same loop.

## Way 2: `model.fit()` — Keras's built-in training loop

Because Keras models already know their loss and optimizer once you
`compile()` them, TensorFlow can offer a one-line training loop that
does everything above for you:

```python
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test))
```

`model.fit()` handles the `GradientTape`/`apply_gradients` machinery,
batching, shuffling, and metric tracking internally, and returns a
`history` object with the loss/accuracy recorded every epoch
(`history.history["loss"]`, `history.history["val_accuracy"]`, etc.).
**Know both approaches**: `fit()` for everyday use, the custom loop for
anything `fit()` doesn't directly support (custom training logic, several
models trained jointly, research-y modifications) — every day from here
uses whichever is clearer for that day's topic.

## `training=True/False` in practice

```python
predictions = model(X_test, training=False)   # inference mode (Day 3)
```

You met this on Day 3 — this is where it matters in practice: always
evaluate with `training=False`, since you're not updating weights and
don't want layers like Dropout (Day 10) behaving as if you were still
training. `model.fit()`/`model.evaluate()` set this for you
automatically; only matters when you write the custom loop yourself.

## Train/test split

A simple shuffle-then-slice split (80% train, 20% test) lets you check
whether the model actually *generalizes* to data it never trained on,
rather than just memorizing the training examples:

```python
indices = tf.random.shuffle(tf.range(len(X)))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]
X_train, Y_train = tf.gather(X, train_idx), tf.gather(Y, train_idx)
X_test, Y_test = tf.gather(X, test_idx), tf.gather(Y, test_idx)
```

This is a preview — Day 10 covers overfitting and proper validation
practice in depth.

## Reading the output

Whichever loop you use, watch the loss decrease and accuracy increase
over training — that's the model's weights being nudged, step by step,
toward values that separate the classes correctly. The final test
accuracy tells you whether it learned the actual *pattern* rather than
just the specific training points.

## Try it yourself

1. Generate three synthetic 2D clusters (three groups of
   `tf.random.normal` points around three different centers) with integer
   labels 0/1/2, split 80/20 train/test.
2. Build a small `tf.keras.Sequential` classifier (a couple of `Dense`
   layers with `relu`, ending in `Dense(3)` — 3 logits, no activation) and
   train it with the **custom loop** for 100 epochs, printing loss every
   10 epochs.
3. Rebuild the exact same architecture, `compile()` it, and train it with
   `model.fit()` instead for the same 100 epochs — compare final loss and
   accuracy against your custom loop's result.
4. Evaluate both trained models on the held-out test set
   (`training=False` for the custom one, `model.evaluate()` for the Keras
   one) and confirm the accuracies roughly agree.
