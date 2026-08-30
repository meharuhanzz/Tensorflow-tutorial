"""TensorFlow Day 8 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import tensorflow as tf

# 1. Build a small CNN Sequential: two conv_blocks (8 then 16 filters)
#    followed by Flatten() and a Dense(3) head, for 32x32x3 inputs. Run
#    model.summary() after one forward call and read off how the spatial
#    size shrinks (32 -> 16 -> 8) through the two pooling layers.
# TODO

# 2. Change the first Conv2D's padding from "same" to "valid" and re-run
#    summary() -- note how the output spatial size differs, and explain
#    why in a comment.
# TODO

# 3. Replace MaxPooling2D in one block with strides=2 directly on the
#    Conv2D layer instead (and drop the pooling layer) -- confirm the
#    output shape after that block is the same either way.
# TODO

# 4. Swap the final Flatten() for GlobalAveragePooling2D() instead, and
#    compare model.count_params() between the two versions -- which has
#    far fewer parameters, and why?
# TODO
