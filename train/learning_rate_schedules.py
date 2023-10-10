"""Implements custom learning rate schedules."""

import tensorflow as tf


class InverseSquareRootDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
    """Implements the learning rate schedule."""

    def __init__(
            self,
            lr_max=1e-3,
            warmup_init_lr=0.0,
            warmup_steps=4000,
            **kwargs):
        super().__init__(**kwargs)
        self._lr_max = lr_max
        self._warmup_init_lr = warmup_init_lr
        self._warmup_steps = warmup_steps

    def __call__(self, step):
        norm_step = step / self._warmup_steps

        def true_fn():
            return (self._warmup_init_lr +
                    (self._lr_max - self._warmup_init_lr) * norm_step)

        def false_fn():
            return self._lr_max * tf.math.rsqrt(norm_step)

        return tf.cond(norm_step <= 1.0, true_fn, false_fn)
