"""TensorFlow Day 12 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import tensorflow as tf

# 1. Build and train a small model for a couple of epochs on random data.
#    Save its weights with save_weights, build a fresh instance of the
#    same architecture, load the weights in, and confirm both models
#    produce identical predictions on the same input.
# TODO

# 2. Save the same trained model with model.save("full_model.keras"),
#    reload it with tf.keras.models.load_model, and confirm its
#    predictions match the original model's.
# TODO

# 3. Set up a tf.train.Checkpoint wrapping both your model and its
#    optimizer, save it mid-training, restore it into a FRESH model +
#    optimizer pair, and confirm training loss continues smoothly from
#    where it left off rather than spiking.
# TODO

# 4. Add a ModelCheckpoint(save_best_only=True) callback to a 30-epoch
#    training run and confirm the saved file corresponds to an earlier,
#    better epoch than the model's final-epoch weights (especially if
#    you deliberately overfit, e.g. with a tiny/noisy dataset).
# TODO
