"""Keras Layers for homology detection from local sequence alignments."""

import tensorflow as tf


class LogCorrectedLogits(tf.keras.layers.Layer):
    """Computes homology detection logits with length correction.

    Logits are computed as
      logits = b + lambda S - K log(len1 * len2).
    """

    def __init__(self,
                 bias_init=tf.initializers.Zeros(),
                 log_lambda_init=tf.initializers.Constant(-1.45),
                 log_k_init=tf.initializers.Constant(-3.03),
                 **kwargs):
        super().__init__(**kwargs)
        self.b = self.add_weight(
            shape=(), initializer=bias_init, name='homology_bias')
        self.log_l = self.add_weight(
            shape=(), initializer=log_lambda_init, name='homology_log_lambda')
        self.log_k = self.add_weight(
            shape=(), initializer=log_k_init, name='homology_log_k')

    def call(self, alignments, mask=None):
        """Computes homology detection logits from SW scores and seq lengths.

        Args:
          alignments: a 2-tuple of scores and paths for the batch.
          mask: a single tf.Tensor<float>[batch, 2, len], corresponding to the
            paddings masks for the two sequences.

        Returns:
          A tf.Tensor<float>[batch, 1] with the logits for each example in the
          batch.
        """
        scores = alignments[0]
        length_fn = lambda x: tf.reduce_sum(x, axis=1)
        masks = tf.cast(mask, tf.float32)
        mn = tf.cast(length_fn(masks[:, 0]) * length_fn(masks[:, 1]), scores.dtype)
        logits = (self.b + tf.exp(self.log_l) * scores -
                  tf.exp(self.log_k) * tf.math.log(mn))
        return logits[:, tf.newaxis]
