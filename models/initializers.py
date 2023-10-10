"""Implements custom initializers."""

import tensorflow as tf


class HarmonicEmbeddings(tf.initializers.Initializer):
    """Initializes weights for sinusoidal positional embeddings.

    Attributes:
      scale_factor: angular frequencies for sinusoidal embeddings will be
        logarithmically spaced between max_freq x scale_factor and max_freq,
        with base equal to scale_factor.
      max_freq: the largest angular frequency to be used.
    """

    def __init__(self, scale_factor=1e-4, max_freq=1.0, **kwargs):
        super().__init__(**kwargs)
        self._scale_factor = scale_factor
        self._max_freq = max_freq

    def __call__(
            self,
            shape,
            dtype=tf.float32,
    ):
        if len(shape) != 2:
            raise ValueError('shape must have length two.')
        max_len, emb_dim = shape
        if emb_dim % 2:
            raise ValueError('dimension of embeddings must be even.')
        n_freqs = emb_dim // 2

        pos = tf.range(max_len, dtype=dtype)
        ang_freq = self._max_freq * tf.experimental.numpy.logspace(
            0.0, 1.0, n_freqs, base=self._scale_factor, dtype=dtype)
        phase = pos[:, None] * ang_freq[None, :]
        embeddings = tf.concat(
            (tf.sin(phase)[:, :, None], tf.cos(phase)[:, :, None]), -1)
        return tf.reshape(embeddings, (max_len, -1))

    def get_config(self):
        config = super().get_config()
        config.update({
            'scale_factor': self._scale_factor,
            'max_freq': self._max_freq,
        })
        return config


class SymmetricKernelInitializer(tf.initializers.Initializer):
    """Initializes 2D symmetric kernel for bilinear form layers.

    Attributes:
      base_init: a Keras initializer to sample the possibly asymmetric kernel.
      factorized: whether to transform the kernel returned by base_init via
        "factorization", W <- W W^{T}, or not, W <- 0.5 (W + W^{T}).
    """

    def __init__(
            self,
            base_init='GlorotUniform',
            factorized=True,
            **kwargs):
        super().__init__(**kwargs)
        self.base_init = tf.keras.initializers.get(base_init)
        self.factorized = factorized

    def __call__(
            self,
            shape,
            dtype=tf.float32,
    ):
        if len(shape) != 2:
            raise ValueError('shape must have length two.')
        kernel = self.base_init(shape, dtype)
        return (tf.matmul(kernel, kernel, transpose_b=True) if self.factorized
                else 0.5 * (kernel + tf.transpose(kernel)))

    def get_config(self):
        config = super().get_config()
        config.update({
            'base_init': tf.keras.initializers.serialize(self.base_init),
            'factorized': self.factorized,
        })
        return config
