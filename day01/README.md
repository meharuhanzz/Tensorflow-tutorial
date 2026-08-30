# TensorFlow Day 1 — Tensors

Everything in TensorFlow is built on the **tensor** — a multi-dimensional
array, similar to a NumPy array, but with two superpowers NumPy doesn't
have: it can track gradients automatically (Day 2), and it can run on a
GPU.

TensorFlow has two closely related tensor types you'll use constantly:

- **`tf.constant`** — an immutable tensor. Once created, its values never
  change.
- **`tf.Variable`** — a mutable tensor, used for anything a model needs
  to *update* during training (weights, biases). Think of `tf.constant`
  as data, `tf.Variable` as parameters.

## Creating tensors

```python
tf.constant([1, 2, 3])      # from a Python list
tf.zeros((2, 3))             # a 2x3 tensor of zeros
tf.ones((2, 3))              # a 2x3 tensor of ones
tf.random.uniform((2, 3))    # random values in [0, 1)
tf.range(12)                  # 0, 1, 2, ..., 11 -- like Python's range()
tf.Variable([1.0, 2.0])      # a mutable tensor
```

## Shape, dtype, dimensions

- `.shape` — the size along each dimension (e.g. `TensorShape([3, 4])`)
- `.dtype` — the data type (`tf.float32`, `tf.int32`, etc.)
- `tf.rank(t)` — how many dimensions (or just `len(t.shape)`)

## Operations: element-wise vs. matrix multiplication

This is the single most important distinction to get right on Day 1:

```python
x * y             # element-wise: multiplies matching positions
x @ y              # matrix multiplication (or tf.matmul(x, y))
```

For `x * y`, both tensors need the same shape (or be "broadcastable" — a
NumPy/TensorFlow rule for compatible shapes). For `x @ y`, the *inner*
dimensions must match — a `(2, 3)` tensor times a `(3, 4)` tensor gives a
`(2, 4)` result. Mixing these two up is one of the most common early
TensorFlow bugs.

## Indexing, slicing, reshaping

Same rules as Python lists and NumPy arrays — `t[0]` for the first row,
`t[:, 0]` for the first column, `t[1, 2]` for a single element.

`tf.reshape(t, shape)` changes a tensor's shape without changing its
data. Passing `-1` for one dimension tells TensorFlow "figure this size
out for me" — `tf.reshape(flat, (2, -1))` on a 12-element tensor gives
you a `(2, 6)` tensor automatically.

## Tensors and NumPy

```python
tf.constant(np_array)     # NumPy -> tensor
tensor.numpy()              # tensor -> NumPy
```

Unlike PyTorch, a TensorFlow tensor created from a NumPy array does
**not** share memory with it — converting makes a copy. Not something to
worry about yet, but worth knowing if you're coming from PyTorch.

## Devices — writing GPU-ready code from day one

```python
gpus = tf.config.list_physical_devices("GPU")
device = "/GPU:0" if gpus else "/CPU:0"
with tf.device(device):
    tensor = tf.random.uniform((2, 2))
```

TensorFlow actually places tensors and ops on a GPU **automatically**
whenever one is visible — you don't strictly need `tf.device(...)` the
way PyTorch needs `.to(device)`. But writing explicit `tf.device(...)`
blocks is still good practice when you want to *force* something onto
CPU or a specific GPU (common in later days), so we introduce the
pattern now.

## Try it yourself

Open a Python shell (or a `.py` file) and work through the examples
above line by line — then try:

1. Create a 1D tensor of the numbers 10 to 20 (inclusive) with `tf.range()`.
2. Create two 3x3 random tensors and compute both their element-wise
   product and their matrix product — confirm the shapes differ.
3. Create `t = tf.range(24)`, reshape it to `(2, 3, 4)`, and index into it
   to grab the single number at position `[1, 2, 3]`.
4. Write device-agnostic code: check `tf.config.list_physical_devices("GPU")`,
   build a device string, and create a tensor inside a `with tf.device(...)`
   block.
