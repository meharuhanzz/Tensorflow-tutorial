# TensorFlow Day 7 — Real Image Data

Days 1-6 used tensors and synthetic 2D points. Today: real image files on
disk, loaded the way every real TensorFlow image project actually works.

## `image_dataset_from_directory` — loading a labeled dataset from folders

The convention (same one PyTorch's `ImageFolder` uses): one subfolder
per class.

```
sample_images/
  red_ish/
    red_ish_0.png
    red_ish_1.png
    ...
  green_ish/
    ...
  blue_ish/
    ...
```

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    "sample_images",
    image_size=(32, 32),
    batch_size=16,
)
```

This single call reads the folder structure, assigns integer labels from
the subfolder names (`train_ds.class_names` gives you the mapping back),
resizes every image, batches them, and returns a ready-to-use
`tf.data.Dataset` (Day 6) — no separate "transform" object needed the way
PyTorch requires, though see below for when you do still want one.

## Rescaling — pixel values to `[0, 1]`

Images load with pixel values in `[0, 255]`. Neural networks train much
better on small numbers:

```python
rescale = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (rescale(x), y))
```

`Rescaling` is a *layer*, not a plain function — that's deliberate (see
below).

## Data augmentation — as model layers, only active during training

TensorFlow's idiom here is genuinely different from PyTorch's
`transforms.Compose`: augmentation is built from ordinary Keras layers,
which you can either `.map()` over your dataset or — more commonly —
attach directly to the *front of your model*:

```python
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
])

model = tf.keras.Sequential([
    augmentation,                 # only active when training=True
    tf.keras.layers.Rescaling(1.0 / 255),
    # ... the rest of your CNN (Day 8) ...
])
```

These augmentation layers check the same `training` flag from Day 3 —
during `model.fit()` they randomly distort each image; during
`model.evaluate()`/`model.predict()` (or any call with
`training=False`) they pass images through unchanged. This means the
**same model object** does the right thing automatically for both
training and evaluation — you don't need PyTorch's separate `train_tfms`
/ `val_tfms` pipelines, because the model itself only augments when
it's actually training. Run the same augmentation layer on the same
image twice during training and you'll get *different* tensors each
time — that randomness is the point.

## Normalizing to a pretrained model's expected input

Day 11's transfer learning uses models pretrained on ImageNet, which
expect a *specific* preprocessing, not just `[0, 1]` scaling:

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
images = preprocess_input(images)   # matches exactly what that model was trained with
```

Every `tf.keras.applications` model ships its own matching
`preprocess_input` function — always use the one paired with the
specific pretrained model you're loading, not a generic ImageNet
mean/std you compute or copy from elsewhere.

## `.cache()` and `.prefetch()` for image pipelines specifically

```python
train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
```

`.cache()` keeps decoded images in memory after the first epoch instead
of re-reading and re-decoding image files from disk every epoch — a
bigger win for images than it was for Day 6's plain arrays, since image
decoding is comparatively expensive. Skip `.cache()` if your dataset is
too large to fit in memory.

## Try it yourself

1. Create a tiny 3-class folder of solid-ish colored squares on disk
   (a handful of small PNGs per class is enough — generate them with
   `PIL.Image.new` in a throwaway script, or draw a few by hand), then
   load it with `image_dataset_from_directory` and print `class_names`
   and one batch's shape.
2. Build an `augmentation` `Sequential` block (flip + rotation) and run
   the *same* image through it twice with `training=True` — confirm the
   two outputs differ. Run it once more with `training=False` and confirm
   it's unchanged from the original.
3. Attach that augmentation block to the front of a tiny placeholder
   model (even a single `Flatten` + `Dense` is fine for this exercise)
   and confirm `model(x, training=False)` behaves deterministically while
   `model(x, training=True)` doesn't.
4. Look up `tf.keras.applications.resnet50.preprocess_input` and compare
   what it does to a `[0, 255]` image against
   `tf.keras.applications.mobilenet_v2.preprocess_input` on the same
   input — are they identical? (They're not — different pretrained
   models were trained with different preprocessing conventions, which
   is exactly why you must match the function to the model.)
