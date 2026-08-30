"""TensorFlow Day 15 -- Capstone exercises. Run make_shapes.py first if
you haven't already. These extend main.py directly rather than starting
fresh -- exactly the kind of work you'd do on a real project after
getting an initial version working.
"""
import tensorflow as tf

# 1. Generate the 4-class shape dataset (circle/square/triangle/star) and
#    load it with image_dataset_from_directory (see main.py for the
#    exact setup).
# TODO

# 2. Build the MobileNetV2-based model, run the two-stage fine-tuning
#    schedule from main.py, and get a full tf.math.confusion_matrix +
#    per-class-accuracy report (Day 14) on the validation set.
# TODO

# 3. Extend the two-stage schedule to THREE stages: head-only -> unfreeze
#    the last 20 layers -> unfreeze the last 60 layers, shrinking the
#    learning rate further at each stage. Does the extra stage help on
#    this small a dataset, or is two stages already enough?
# TODO

# 4. Add mixed precision (tf.keras.mixed_precision.set_global_policy(
#    "mixed_bfloat16"), Day 13) to the whole pipeline and confirm the
#    model still trains to a comparable final accuracy.
# TODO
