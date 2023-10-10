"""Contains auxiliary functions to operator over "paired" tensors."""

import tensorflow as tf


def pair_masks(mask_x, mask_y):
    """Combines a pair of 2D masks into a single 3D mask.

    Args:
      mask_x: A tf.Tensor<float>[batch, len_x] with binary entries.
      mask_y: A tf.Tensor<float>[batch, len_y] with binary entries.

    Returns:
      A tf.Tensor<float>[batch, len_x, len_y] with binary entries, defined as
        out[n][i][j] := mask_x[n][i] * mask_y[n][j].
    """
    mask1, mask2 = tf.cast(mask_x, tf.float32), tf.cast(mask_y, tf.float32)
    return tf.cast(tf.einsum('ij,ik->ijk', mask1, mask2), tf.bool)


def build(indices, *args):
    """Builds the pairs of whatever is passed as args for the given indices.

    Args:
      indices: a tf.Tensor<int32>[batch, 2]
      *args: a sequence of tf.Tensor[2 * batch, ...].

    Returns:
      A tuple of tf.Tensor[batch, 2, ...]
    """
    return tuple(tf.gather(arg, indices) for arg in args)


def consecutive_indices(batch):
    """Builds a batch of consecutive indices of size N from a batch of size 2N.

    Args:
      batch: tf.Tensor<float>[2N, ...].

    Returns:
      A tf.Tensor<int32>[N, 2] of consecutive indices.
    """
    batch_size = tf.shape(batch)[0]
    return tf.reshape(tf.range(batch_size), (-1, 2))
