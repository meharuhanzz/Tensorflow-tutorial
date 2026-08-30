"""TensorFlow Day 13 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import tensorflow as tf

# 1. Print tf.config.list_physical_devices("GPU") on your machine and
#    note whether you have one.
# TODO

# 2. Set the global policy to "mixed_bfloat16", build and train a small
#    model for a few epochs on random data, and print
#    model.layers[0].dtype_policy to confirm it picked up the policy.
# TODO

# 3. Check model.dtype (the storage dtype, still float32) against the
#    dtype of an actual forward-pass output (model(x).dtype) under a
#    mixed-precision policy -- confirm they differ, and explain why in a
#    comment.
# TODO

# 4. Train the identical model architecture once under "float32" and once
#    under "mixed_bfloat16" (same data, same seed) and compare final
#    training loss between the two runs.
# TODO
