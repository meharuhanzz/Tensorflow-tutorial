"""TensorFlow Day 3 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import tensorflow as tf

# 1. Build a SimpleNet-style subclass with three Dense layers (8 -> 8 -> 1)
#    and ReLU activations between the first two. Call it once on
#    tf.random.uniform((5, 4)) and print the output shape.
# TODO

# 2. Call model.summary() before and after that first call -- confirm it
#    only reports real parameter counts AFTER the model has seen input.
# TODO

# 3. Print model.count_params() and manually verify it against the layer
#    sizes you chose (each Dense(n) layer contributes
#    input_dim * n + n parameters -- the "+ n" is the bias).
# TODO

# 4. Rebuild the same architecture as a tf.keras.Sequential instead, and
#    confirm count_params() matches your subclassed version.
# TODO
