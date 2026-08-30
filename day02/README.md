# TensorFlow Day 2 — `tf.GradientTape`

Automatic differentiation is the feature that makes TensorFlow a *deep
learning* framework rather than just "NumPy with tensors" — it computes
derivatives for you, which is exactly what training a neural network
needs (backpropagation is just repeated derivative computation).

## What gets tracked, automatically

```python
x = tf.Variable(3.0)
```

A `tf.Variable` is watched **automatically** by any `GradientTape` it's
used inside — this is the one thing to internalize on Day 2. You'll make
your model's learnable weights `tf.Variable`s (Keras does this for you
starting Day 3) — not your input data, which stays as plain tensors.

## `tf.GradientTape` and `.gradient()`

```python
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2

dy_dx = tape.gradient(y, x)
print(dy_dx)   # dy/dx = 2x -- evaluated at x's current value
```

Everything computed *inside* the `with` block is recorded onto a
"computation graph" (a tape, literally). `tape.gradient(y, x)` walks that
tape backward and returns the derivative — you never write the
derivative formula yourself.

## Watching a plain tensor with `tape.watch()`

Only `tf.Variable`s are tracked automatically. If you need the gradient
with respect to a plain `tf.constant`, tell the tape explicitly:

```python
x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2
dy_dx = tape.gradient(y, x)
```

## The #1 early gotcha: a tape is single-use

```python
with tf.GradientTape() as tape:
    y = x ** 2

tape.gradient(y, x)   # fine
tape.gradient(y, x)   # RuntimeError! the tape has already been "consumed"
```

Unlike PyTorch (where `.grad` accumulates silently across multiple
`.backward()` calls unless you zero it), TensorFlow fails loudly here —
by default, `tape.gradient()` releases the tape's resources the moment
you call it once. If you genuinely need multiple gradients from the same
forward pass, open the tape with `persistent=True` and `del tape`
yourself when done:

```python
with tf.GradientTape(persistent=True) as tape:
    y = x ** 2
    z = x ** 3
dy_dx = tape.gradient(y, x)
dz_dx = tape.gradient(z, x)   # works now
del tape   # release resources when you're finished
```

## No `torch.no_grad()` needed — just don't open a tape

In PyTorch you have to explicitly wrap inference code in
`torch.no_grad()` to avoid wasting memory tracking gradients you don't
need. In TensorFlow there's nothing to opt out of — if code isn't inside
a `with tf.GradientTape():` block, no graph is built at all. Simpler, but
worth knowing explicitly since it's a real difference between the two
frameworks.

## `tf.stop_gradient()` — PyTorch's `.detach()`

```python
y_stopped = tf.stop_gradient(y)
```

Gives you a tensor with the same value, but gradients won't flow back
through it — useful when part of a computation should be treated as a
constant during backprop.

## Manual gradient descent — the core training idea, in miniature

```python
w = tf.Variable(0.0)
for step in range(5):
    with tf.GradientTape() as tape:
        loss = (w - 10) ** 2
    grad = tape.gradient(loss, w)
    w.assign_sub(0.1 * grad)   # w -= 0.1 * grad, but as a Variable op
```

Every neural network training loop is a generalization of exactly this:
compute a loss inside a tape, get the gradient, nudge every weight a
little in the direction that reduces the loss, repeat. Day 5 turns this
into the real thing using `tf.keras.Model` (Day 3) and an optimizer
(Day 4) instead of doing the weight update by hand.

## Try it yourself

1. Create `x = tf.Variable(5.0)`. Inside a tape, compute `y = x**3 - 2*x`.
   Get `tape.gradient(y, x)` and check it against the hand-derivative
   `3x^2 - 2` at `x=5`.
2. Create `a = tf.Variable(1.0)`, `b = tf.Variable(4.0)`. Inside one tape,
   compute `z = a * b + b**2`, then get both gradients in a single
   `tape.gradient(z, [a, b])` call.
3. Create a plain `tf.constant(2.0)` and try differentiating `c**2`
   *without* `tape.watch(c)` first — see what you get back — then add
   `tape.watch(c)` and compare.
4. Write a 10-step manual gradient descent loop minimizing `(w + 3) ** 2`
   starting from `w = tf.Variable(0.0)`, `learning_rate = 0.1`. What value
   does `w` converge to?
