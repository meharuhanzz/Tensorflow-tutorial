# TensorFlow Day 9 — Training a CNN End-to-End

Everything from Days 6-8 comes together into one real training run:
`image_dataset_from_directory` (Day 7) feeding a CNN (Day 8) through a
full training loop with validation (Day 5).

Unlike Day 7's color-only sample images, a genuinely useful test dataset
for this needs to require learning *shape*, not just color — three
classes of simple shapes (circles/squares/triangles) with randomized
position, size, color, and a noisy background is a good, honest test of
whether the CNN is learning something real rather than shortcutting on
a single pixel value.

## `validation_split` — an easier train/val split

Day 5/6 split data by hand. `image_dataset_from_directory` does the same
job for a folder of images in two calls, one `subset="training"` and one
`subset="validation"` — both need the *same* `seed` so they split the
files consistently:

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    "shapes", validation_split=0.2, subset="training", seed=42,
    image_size=(32, 32), batch_size=16)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "shapes", validation_split=0.2, subset="validation", seed=42,
    image_size=(32, 32), batch_size=16)
```

## The full training script, structurally

**With `model.fit()`** (Day 5's "Way 2"), this whole day collapses to
one call:

```python
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
history = model.fit(train_ds, validation_data=val_ds, epochs=15)
```

**With the custom loop** (Day 5's "Way 1"), splitting training and
evaluation into their own functions is standard practice once a script
grows past a few lines — it also makes the difference between the two
crystal clear:

```python
def train_one_epoch(train_ds):
    for x, y in train_ds:
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

def evaluate(val_ds):
    correct, total = 0, 0
    for x, y in val_ds:
        logits = model(x, training=False)   # no tape at all -- Day 2
        preds = tf.argmax(logits, axis=1)
        correct += tf.reduce_sum(tf.cast(preds == tf.cast(y, preds.dtype), tf.int32))
        total += len(y)
    return correct / total

for epoch in range(num_epochs):
    train_one_epoch(train_ds)
    val_acc = evaluate(val_ds)
    print(f"epoch {epoch}: val_acc = {val_acc:.3f}")
```

Training computes gradients and updates weights (tape + `apply_gradients`,
Day 2/4); evaluation does neither — no tape at all, no optimizer calls.

## Reading the output

Watch **both** train and validation accuracy across epochs (`history.history["accuracy"]`
vs. `history.history["val_accuracy"]` if using `fit()`), not just train.
If training accuracy climbs but validation accuracy stalls or drops,
that's **overfitting** — the model memorizing training examples rather
than learning the general pattern. Day 10 covers this properly; for now,
just get used to looking at both numbers side by side.

## Why this matters beyond toy shapes

This exact structure — `image_dataset_from_directory`, a CNN,
`SparseCategoricalCrossentropy` + `Adam`, a train/val loop — is the same
fundamental loop behind any real Keras image classifier. The only real
differences in a production version: a much bigger, ImageNet-pretrained
model instead of training from scratch (Day 11 covers exactly this), and
a harder, real dataset instead of synthetic shapes.

## Try it yourself

1. Generate (or hand-draw with `PIL.ImageDraw`) a tiny 3-class shape
   dataset — a handful of circles/squares/triangles per class, randomized
   position and color — and load it with `image_dataset_from_directory`
   using `validation_split=0.2`.
2. Build the Day 8 CNN architecture, `compile()` it, and train with
   `model.fit()` for 15 epochs. Plot or print `history.history["accuracy"]`
   vs `history.history["val_accuracy"]` side by side.
3. Reimplement the same training run with the custom loop instead
   (`train_one_epoch` / `evaluate` functions above) and confirm you get a
   comparable final validation accuracy.
4. Deliberately shrink the training set to just 2-3 images per class and
   retrain — watch train accuracy shoot to 100% while validation accuracy
   stays low or noisy. This is overfitting, previewed here, covered
   properly on Day 10.
