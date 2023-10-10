"""Implements custom activation functions."""

import tensorflow as tf


def approximate_gelu(x):
    return tf.nn.gelu(x, True)
