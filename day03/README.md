# TensorFlow Day 3 — Building a Model with `tf.keras.Model`

## `tf.keras.layers.Dense` — a fully connected layer

```python
layer = tf.keras.layers.Dense(units=1)
output = layer(x)   # computes x @ kernel + bias
```

Creating a `Dense` layer doesn't create its weights right away — they're
created the *first time* the layer actually sees input, once TensorFlow
knows the input size (this is called "lazy building," and it's a real
difference from PyTorch's `nn.Linear`, which needs `in_features` up
front). You never manually create the weights yourself either way.

## The standard pattern: subclassing `tf.keras.Model`

Virtually every non-trivial Keras model follows this exact shape — it
will look immediately familiar if you know PyTorch's `nn.Module`:

```python
class SimpleNet(tf.keras.Model):
    def __init__(self):
        super().__init__()          # always call this first
        self.layer1 = tf.keras.layers.Dense(8)
        self.layer2 = tf.keras.layers.Dense(3)
        self.activation = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
```

- **`__init__`** — declare the layers you'll use.
- **`call`** — describe how data actually flows through those layers
  (this is Keras's name for what PyTorch calls `forward`).
- Calling `model(x)` runs `model.call(x)` for you — don't call `.call()`
  directly, same rule as PyTorch's `.forward()`.

## Why non-linear activations matter

Stack two `Dense` layers with *nothing* between them, and mathematically
it's equivalent to just one bigger `Dense` layer — no matter how many
layers you add, you'd still only be able to represent straight-line
(linear) relationships. Inserting a non-linear function like `ReLU`
between layers is what lets a deep stack of layers represent genuinely
complex, curved functions. `ReLU` itself is simple: negative numbers
become 0, positive numbers pass through unchanged.

## Inspecting a model

```python
model.build(input_shape=(None, 4))            # force weight creation, see below
model.count_params()                             # total parameter count
model.summary()                                   # a readable table of every layer
model.trainable_variables                         # list of every tf.Variable
```

Every one of those `trainable_variable`s is exactly what Day 2's
`GradientTape` machinery will differentiate through, once there's a loss
function (Day 4) to compute a gradient *of*.

**On lazy building**: since layers only build their weights on first
call, `model.summary()` before that first call shows every layer as
`(unbuilt)` with 0 params — not wrong, just unhelpful, and an easy trap
if you forget why your freshly-built model reports zero parameters.
Either call the model once on a dummy input (`model(tf.zeros((1, 4)))`)
or call `model.build(input_shape=...)` explicitly first, *then* inspect it.

## `tf.keras.Sequential` — a shortcut for simple pipelines

When your model is just "run these layers one after another, nothing
fancier," `Sequential` saves you writing an explicit class:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(2),
])
```

Notice `activation="relu"` can be passed straight into `Dense` — a
convenience PyTorch's `nn.Linear` doesn't have (there you always chain a
separate `nn.ReLU()`). Reach for a full `tf.keras.Model` subclass instead
of `Sequential` once you need branching, skip connections, or any logic
beyond a straight line of layers.

## `training=True/False` — Keras's version of `train()`/`eval()`

```python
output = model(x, training=True)     # training-mode behavior
output = model(x, training=False)    # inference-mode behavior (the default)
```

Rather than a persistent mode flag you toggle on the model object (like
PyTorch's `model.train()` / `model.eval()`), Keras layers take a
`training` argument on every call. Specific layer types (Dropout,
BatchNormalization — met properly on Day 10) check this argument to
behave differently during training vs. inference. Keras's built-in
`model.fit()` (Day 5) sets this for you automatically; it only matters
when you write your own training loop.

## Try it yourself

1. Build a `SimpleNet`-style subclass with three `Dense` layers (8 → 8 → 1)
   and `ReLU` activations between the first two. Call it once on
   `tf.random.uniform((5, 4))` and print the output shape.
2. Call `model.summary()` before and after that first call — confirm it
   only works *after* the model has seen input once.
3. Print `model.count_params()` and manually verify it against the
   layer sizes you chose (remember: each `Dense(n)` layer contributes
   `input_dim * n + n` parameters — the `+ n` is the bias).
4. Rebuild the same architecture as a `tf.keras.Sequential` instead, and
   confirm `count_params()` matches your subclassed version.
