"""TensorFlow Day 10 -- Exercises. Run make_shapes.py first if you
haven't already (or reuse Day 9's if you're working through in order).
Fill in the TODOs, then run: python3 exercises.py
"""
import tensorflow as tf

# 1. Train Day 9's CNN on a deliberately tiny training set (2-3 images
#    per class) for 40 epochs with no regularization at all. Plot or
#    print train vs. validation accuracy per epoch and confirm you see
#    the overfitting gap.
# TODO

# 2. Add a Dropout(0.5) layer right before the final Dense classifier and
#    retrain from scratch on the same tiny dataset -- compare the gap.
# TODO

# 3. Instead (or as well), add kernel_regularizer=tf.keras.regularizers.l2(1e-3)
#    to your Dense layers and retrain -- compare against both previous
#    runs.
# TODO

# 4. Wrap a longer (100-epoch) training run in EarlyStopping(monitor="val_loss",
#    patience=5, restore_best_weights=True) and confirm training actually
#    stops before reaching epoch 100 once validation loss stalls.
# TODO
