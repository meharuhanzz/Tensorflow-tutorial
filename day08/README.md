# TensorFlow Day 8 — Convolutional Neural Networks

## Why not just flatten the image and use Dense layers?

You could — but you'd throw away something important: **which pixels are
near each other**. A `Dense` layer treats every pixel as equally
(un)related to every other pixel, with no notion that nearby pixels often
form an edge, a texture, or a shape. Convolutional layers are built
specifically to exploit that spatial structure.

## `tf.keras.layers.Conv2D` — a small filter that slides across the image

```python
conv = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding="same", activation="relu")
```

- **`filters`** — how many different filters to learn (PyTorch calls this
  `out_channels`). Each one produces its own output "feature map" (e.g.
  one might learn to detect vertical edges, another horizontal edges,
  another a particular color transition). The input channel count
  (`in_channels` in PyTorch) is inferred automatically from whatever
  data the layer first sees — another instance of Day 3's lazy building.
- **`kernel_size`** — the filter's size (`3` means a 3x3 window).
- **`padding="same"`** — pads the image edges so the output keeps the
  same height/width as the input (the alternative, `padding="valid"`,
  is PyTorch's default — no padding, output slightly shrinks).
- **`activation="relu"`** — Keras lets you fuse the activation straight
  into the conv layer, the same convenience you saw with `Dense` on
  Day 3, instead of chaining a separate `ReLU()` layer.

The key idea: the **same** learned filter is applied at every position in
the image — this is what lets a `Conv2D` layer detect "an edge" wherever
it appears, rather than only in one specific spot (which is what would
happen with a `Dense` layer).

## `strides` — downsampling as you convolve

```python
tf.keras.layers.Conv2D(8, kernel_size=3, padding="same", strides=2)
```

`strides=2` moves the filter 2 pixels at a time instead of 1, roughly
halving the output's height and width. This is one common way to shrink
the feature map as the network goes deeper.

## `tf.keras.layers.MaxPooling2D` — another way to downsample

```python
tf.keras.layers.MaxPooling2D(pool_size=2)   # keeps only the max value in each 2x2 window
```

Pooling has no learnable parameters — it's a fixed operation that shrinks
the feature map and adds a bit of robustness to small shifts in the
image (an edge detected slightly off-position still gets picked up).

## Channel order: TensorFlow's default is different from PyTorch's

PyTorch image tensors are `(batch, channels, H, W)` — "channels-first."
TensorFlow's default is `(batch, H, W, channels)` — "channels-last."
Both are valid conventions and TensorFlow *can* be configured for
channels-first, but channels-last is the default and what virtually all
Keras code (including everything in this course) assumes. Worth knowing
explicitly the first time you see a shape like `(32, 32, 3)` instead of
`(3, 32, 32)` and wonder why the 3 moved.

## The classic CNN block

```python
conv_block = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
])
```

Stack several of these, and a common pattern emerges: each block
*shrinks* the height/width (via pooling) while *growing* the channel
count (more filters) — the network trades spatial resolution for a
richer set of learned features as it goes deeper.

## `Flatten` — the bridge to `Dense` layers

```python
tf.keras.layers.Flatten()   # (batch, H, W, channels) -> (batch, H*W*channels)
```

Conv/pool layers work on 4D tensors; `Dense` layers expect 2D
`(batch, features)` input. `Flatten` is the standard way to convert
between the two — always the last step of a CNN's "feature extractor"
before its final classification layers. (`GlobalAveragePooling2D` is a
common, often better, alternative you'll meet properly on Day 11 —
averages each feature map down to one number instead of keeping every
spatial position.)

## Try it yourself

1. Build a small CNN `Sequential`: two `conv_block`s (8 then 16 filters)
   followed by `Flatten()` and a `Dense(3)` head, for 32x32x3 inputs.
   Run `model.summary()` after one forward call and read off how the
   spatial size shrinks (32 → 16 → 8) through the two pooling layers.
2. Change the first `Conv2D`'s `padding` from `"same"` to `"valid"` and
   re-run `summary()` — note how the output spatial size differs, and
   explain why in a comment.
3. Replace `MaxPooling2D` in one block with `strides=2` directly on the
   `Conv2D` layer instead (and drop the pooling layer) — confirm the
   output shape after that block is the same either way.
4. Swap the final `Flatten()` for `GlobalAveragePooling2D()` instead, and
   compare `model.count_params()` between the two versions — which has
   far fewer parameters, and why (hint: think about what `Flatten` does
   to a large H*W right before a `Dense` layer)?
