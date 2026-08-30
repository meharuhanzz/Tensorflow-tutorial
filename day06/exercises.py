"""TensorFlow Day 6 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import tensorflow as tf

# 1. Build a tf.data.Dataset from 100 random (x, y) pairs using
#    from_tensor_slices, then iterate over it with a plain for loop
#    printing just the first 3 examples (no batching yet).
# TODO

# 2. Add .shuffle(100).batch(16) and iterate again -- print the shape of
#    each batch and confirm the last batch is smaller than 16 (100 isn't
#    evenly divisible by 16).
# TODO

# 3. Reuse Day 5's three-cluster classification data and training loop,
#    but swap the whole-dataset-at-once forward pass for a proper
#    tf.data.Dataset pipeline (shuffle + .batch(16) + .prefetch),
#    training for 20 epochs.
# TODO

# 4. Explain in a comment: what would go wrong if you called .batch()
#    BEFORE .shuffle() instead of after?
# TODO
