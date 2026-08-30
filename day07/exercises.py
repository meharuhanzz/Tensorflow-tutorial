"""TensorFlow Day 7 -- Exercises. Run make_sample_images.py first if you
haven't already. Fill in the TODOs, then run: python3 exercises.py
"""
import tensorflow as tf

# 1. Load sample_images/ with image_dataset_from_directory and print
#    class_names and one batch's shape.
# TODO

# 2. Build an augmentation Sequential block (flip + rotation) and run the
#    SAME image through it twice with training=True -- confirm the two
#    outputs differ. Run it once more with training=False and confirm
#    it's unchanged from the original.
# TODO

# 3. Attach that augmentation block to the front of a tiny placeholder
#    model (even a single Flatten + Dense is fine) and confirm
#    model(x, training=False) behaves deterministically while
#    model(x, training=True) doesn't.
# TODO

# 4. Look up tf.keras.applications.resnet50.preprocess_input and compare
#    what it does to a [0, 255] image against
#    tf.keras.applications.mobilenet_v2.preprocess_input on the same
#    input -- are they identical?
# TODO
