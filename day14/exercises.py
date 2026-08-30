"""TensorFlow Day 14 -- Exercises. Run make_shapes.py first if you
haven't already. Fill in the TODOs, then run: python3 exercises.py
"""
import tensorflow as tf

# 1. Train Day 9's CNN on a 3-class shape dataset, collect
#    all_preds/all_labels over the validation set, and print
#    tf.math.confusion_matrix(all_labels, all_preds).
# TODO

# 2. Compute per-class accuracy from that confusion matrix and identify
#    which class (if any) the model does worst on.
# TODO

# 3. Compute precision and recall per class using tf.keras.metrics.Precision/
#    Recall with a class_id argument -- remember BOTH y_true and y_pred
#    need to be one-hot / per-class-score arrays here (tf.one_hot the
#    labels, tf.nn.softmax the logits), not sparse integers or argmaxed
#    predictions. Compare against the confusion matrix's per-class
#    accuracy -- same ballpark, not an exact match (different threshold
#    rule: 0.5-per-class vs. argmax).
# TODO

# 4. Pick the 5 validation examples the model got MOST CONFIDENTLY WRONG
#    (highest softmax probability on an incorrect class) and inspect
#    them -- is there a pattern?
# TODO
