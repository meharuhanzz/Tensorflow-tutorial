"""TensorFlow Day 2 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import tensorflow as tf

# 1. Create x = tf.Variable(5.0). Inside a tf.GradientTape() block, compute
#    y = x**3 - 2*x. Get tape.gradient(y, x) and print it. By hand: dy/dx
#    = 3x^2 - 2 -- check your printed value matches at x=5.
# TODO

# 2. Create a = tf.Variable(1.0) and b = tf.Variable(4.0). Inside one tape,
#    compute z = a * b + b**2. Get gradients w.r.t. both [a, b] in one
#    tape.gradient() call and print both.
# TODO

# 3. Create a plain tf.constant(2.0) called c. Try computing its gradient
#    through c**2 WITHOUT calling tape.watch(c) first -- print what
#    tape.gradient() returns (hint: it won't be an error, but it also
#    won't be the answer you expect). Then add tape.watch(c) and compare.
# TODO

# 4. Write a 10-step manual gradient descent loop that minimizes
#    (w + 3) ** 2 starting from w = tf.Variable(0.0), learning_rate = 0.1.
#    Print w and loss every step. What value does w converge to?
# TODO
