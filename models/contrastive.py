import tensorflow as tf


class Contrastive(tf.keras.Model):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    @tf.function
    def call(self, inputs, training=False):
        scores = []
        batch = inputs.shape[0]
        for i in range(batch):
            for j in range(batch):
                # 成对序列的比对分数矩阵是对称的
                if i > j:
                    scores.append(scores[j * batch + i])
                    continue
                mini_input = tf.stack([inputs[i], inputs[j]], 0)
                alignments = self.model(mini_input, training=training)
                scores.append(alignments['sw_scores'])
        scores = tf.convert_to_tensor(scores)
        # 转换为方阵形式
        scores = tf.reshape(scores, (batch, batch))
        return scores
