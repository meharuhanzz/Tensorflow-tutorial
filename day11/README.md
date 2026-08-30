# TensorFlow Day 11 — Transfer Learning

Training a large model from scratch needs enormous amounts of data and
compute. **Transfer learning** sidesteps this: start from a model already
trained on millions of images, and adapt it to your specific, much
smaller task.

## Loading a pretrained model

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,     # drop the original 1000-class ImageNet head
    weights="imagenet",
)
```

This model was already trained on ImageNet (1.2 million images, 1000
classes) — it already knows how to detect edges, textures, and shapes,
purely from that prior training. `include_top=False` strips off the
final classification layers, leaving just the pretrained feature
extractor — you'll attach your own head next.

## Building your own head on top

```python
base_model.trainable = False   # freeze -- see below

inputs = tf.keras.Input(shape=(96, 96, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)   # Day 7
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)   # Day 8
outputs = tf.keras.layers.Dense(num_classes)(x)
model = tf.keras.Model(inputs, outputs)
```

This is Keras's **Functional API** — instead of subclassing (Day 3), you
build a model by calling layers on tensors and wiring the data flow
explicitly. It's the natural style once you're composing a pretrained
model with new layers like this. Note `base_model(x, training=False)`
even though the *outer* model will later be trained — this keeps
BatchNormalization layers inside `base_model` in inference mode, which
matters for fine-tuning stability (more on this below).

## Freezing — feature extraction mode

```python
base_model.trainable = False
```

Setting `.trainable = False` on the whole base model means none of its
layers' weights will be updated during training — it's used purely as a
fixed feature extractor, and only the new `Dense` head (which is
`trainable` by default) actually learns. This is fast and works well
when your images are broadly similar in kind to ImageNet (real-world
photos of physical objects).

```python
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5)
```

## Why transfer learning needs so few epochs

5 epochs training just the head already reaches solid accuracy — compare
that to Day 9/10's from-scratch CNN on similar-sized data, which needed
far more epochs for far less accuracy. The pretrained features are
already doing most of the work; the new head just needs to learn how to
combine them for your specific classes.

## Staged fine-tuning — unfreezing progressively

```python
base_model.trainable = True
for layer in base_model.layers[:-20]:   # keep all but the last 20 layers frozen
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),   # much smaller LR
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=5)
```

Once the new head is reasonably trained, unfreezing more of the network
(the *last* layers, closest to the output) lets the model adapt its
highest-level features to your specific task. Use a **much smaller**
learning rate once more is unfrozen — those layers already have good,
pretrained weights, and large updates would wreck them rather than
gently adapt them. (`1e-5` here vs. the default `1e-3`-ish scale used for
head-only training — roughly two orders of magnitude smaller is typical.)

**A BatchNorm gotcha worth knowing**: even after `base_model.trainable = True`,
BatchNormalization layers inside the base model should generally still
run in inference mode (their moving statistics stay frozen) — this is
exactly why the model call used `training=False` on the base model above,
independent of whether the outer `model.fit()` is in training mode. Get
this wrong and fine-tuning can quietly destabilize a model that was
working fine as a frozen feature extractor.

This two-stage recipe — head-only → unfreeze the last N layers with a
smaller LR — generalizes to a three-or-more-stage version (unfreezing
progressively more, shrinking the LR further each time) for harder
transfer-learning tasks.

## Run it

```bash
python3 make_shapes.py   # once
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
