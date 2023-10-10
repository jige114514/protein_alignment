import tensorflow as tf
import numpy as np


class SimCSE(tf.keras.Model):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    @tf.function
    def call(self, inputs, training=False):
        # inputs = tf.repeat(inputs, 2, 0)
        # pairs = []
        scores = []
        batch = inputs.shape[0]
        # for i in range(batch):
        #     for j in range(batch):
        #         pairs.append(inputs[i])
        #         pairs.append(inputs[j])
        # inputs = tf.stack(pairs, 0)
        # alignments = self.model(inputs, training=training)
        # scores = alignments['sw_scores']
        # scores = tf.reshape(scores, (batch, -1))
        for i in range(batch):
            for j in range(batch):
                if i > j:
                    scores.append(scores[j * batch + i])
                    continue
                mini_input = tf.stack([inputs[i], inputs[j]], 0)
                alignments = self.model(mini_input, training=training)
                scores.append(alignments['sw_scores'])
                # del alignments
        scores = tf.convert_to_tensor(scores)
        scores = tf.reshape(scores, (batch, batch))
        return scores
