"""TensorFlow Day 4 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import tensorflow as tf

# 1. Compute MeanSquaredError()(targets, predictions) for
#    targets = [1.0, 2.0, 3.0] and two different predictions arrays -- one
#    close to targets, one far off. Confirm the loss is bigger for the
#    worse predictions.
# TODO

# 2. Build 3 fake "logits" for a 4-class problem -- one confidently
#    correct, one confidently WRONG, one unconfident/flat -- and compute
#    SparseCategoricalCrossentropy(from_logits=True) for each against the
#    same true label. Order them from lowest to highest loss and confirm
#    it matches your intuition.
# TODO

# 3. Repeat exercise 2 but set from_logits=False by mistake (feeding the
#    same raw logits in) -- observe how the loss values change and why
#    that's a bug, not a feature.
# TODO

# 4. Build a tiny model (a couple of Dense layers), run one manual
#    training step by hand (tape, gradients, apply_gradients) against a
#    single batch of random data and a MeanSquaredError loss against
#    random targets. Print the loss before and after the step and
#    confirm it went down.
# TODO
