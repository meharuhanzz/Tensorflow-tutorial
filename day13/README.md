# TensorFlow Day 13 — GPU & Mixed-Precision Training

**Note on this lesson:** the actual *speedup* from mixed precision is
GPU-specific (it relies on tensor cores). If you're running this on a
machine without a GPU, the code patterns below are still 100% correct
and will run — you just won't see the real performance benefit until you
run the same code on suitable GPU hardware. Focus on understanding the
pattern.

## Device-agnostic code (recap) — and TensorFlow's default is more automatic

```python
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs visible: {len(gpus)}")
```

You met `tf.config.list_physical_devices("GPU")` and `tf.device(...)` on
Day 1. Worth restating here because it matters more today: TensorFlow
places variables and ops on a visible GPU **automatically** by default —
there's no equivalent of PyTorch's `model.to(device)` you must remember
to call. `tf.device(...)` blocks (Day 1) are for the cases where you want
to *override* that default (force something onto CPU, or pick a specific
GPU on a multi-GPU machine).

## Why GPUs matter here

A GPU has thousands of small cores built for exactly the kind of math
neural networks do constantly: large matrix multiplications, run in
parallel. That parallelism is the entire reason a GPU speeds up training
at all.

## Precision: float32 vs float16/bfloat16

TensorFlow tensors default to `float32` — 32 bits of precision per
number. `float16` and `bfloat16` use only 16 bits: half the memory, and
GPUs with tensor cores can multiply matrices in these formats
significantly faster than in `float32`. The tradeoff is reduced numeric
precision — usually fine for deep learning, since neural networks tend
to tolerate small numeric errors well.

## `tf.keras.mixed_precision` — a global policy, not a context manager

This is the biggest structural difference from PyTorch's `torch.autocast`.
Instead of wrapping each forward pass in a `with` block, you set a policy
*once*, globally, before building your model:

```python
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

model = tf.keras.Sequential([...])   # built AFTER the policy is set
model.compile(optimizer="adam", loss=..., metrics=["accuracy"])
model.fit(train_ds, epochs=10)
```

Every layer created after `set_global_policy(...)` automatically computes
in the reduced-precision dtype while keeping its stored weights in
`float32` — "mixed" precision, same underlying idea as PyTorch's
autocast, just configured once instead of wrapped around every step.

## Loss scaling — handled for you with `LossScaleOptimizer`

`float16` has a much smaller exponent range than `float32`, so gradients
can sometimes underflow to zero during backpropagation, silently
stalling training — the same problem PyTorch's `GradScaler` solves.

```python
tf.keras.mixed_precision.set_global_policy("mixed_float16")
optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)   # only for float16
```

If you use `model.compile()` + `model.fit()`, Keras wraps your optimizer
in a `LossScaleOptimizer` **automatically** whenever the global policy is
`"mixed_float16"` — you don't need the line above unless you're writing a
custom `GradientTape` training loop yourself, in which case you must
scale the loss up before `tape.gradient()` and scale the gradients back
down before `apply_gradients()`, mirroring PyTorch's manual
`scaler.scale()` / `scaler.step()` pattern.

`bfloat16` keeps `float32`'s exponent range (just with less mantissa
precision), so it doesn't have this underflow problem — no loss scaling
needed at all, which is why `"mixed_bfloat16"` is the simpler policy to
reach for first, and runs identically on CPU and GPU (useful for
developing and testing this exact lesson without a GPU on hand).

## Why bother

On supported GPU hardware, mixed precision typically gives faster
training (often 1.5-3x) and roughly half the memory usage for
activations held in reduced precision — usually with little to no
accuracy cost. It's close to "free" performance, which is why
`set_global_policy(...)` is one of the first lines in most modern,
performance-conscious Keras training scripts.

## Run it

```bash
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
