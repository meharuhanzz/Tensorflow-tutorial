"""TensorFlow Day 11 -- Exercises. Run make_shapes.py first if you
haven't already. Fill in the TODOs, then run: python3 exercises.py
"""
import tensorflow as tf

# 1. Load MobileNetV2(input_shape=(96,96,3), include_top=False,
#    weights="imagenet"), freeze it, and print len(base_model.layers) and
#    model.count_params() vs. the number of TRAINABLE parameters.
# TODO

# 2. Build the head-only model above for the 3-class shapes problem and
#    train it for 5 epochs. Record final validation accuracy.
# TODO

# 3. Unfreeze the last 20 layers, recompile with a 100x smaller learning
#    rate, and continue training for 5 more epochs -- did validation
#    accuracy improve further?
# TODO

# 4. Try unfreezing the ENTIRE base model with the same small LR
#    (skipping the staged approach) and compare against exercise 3's
#    staged result.
# TODO
