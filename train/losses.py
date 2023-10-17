"""Implements custom losses."""

import tensorflow as tf


# Todo: the contrastive learning loss
class ContrastiveLoss(tf.losses.Loss):
    """Implements a loss for contrastive learning."""

    def __init__(self,
                 name='contrastive_loss',
                 reduction=tf.losses.Reduction.AUTO,
                 temp=0.05):
        """Loss for alignments scores.

        Args:
          name: the name of the loss
          temp: temperature

        Returns:
          A loss function
        """
        self._temp = temp
        super().__init__(name=name, reduction=reduction)

    def call(self, scores, labels):
        """Computes the Contrastive loss.

        Args:
          scores: pos and neg pairs alignments scores
          labels: label which sequence pairs are homologous

        Returns:
          loss
        """
        # 单位对角矩阵——对角线上为1e12很大的值
        c = tf.eye(labels.shape[0]) * 1e12
        # 归一化
        scores = tf.nn.l2_normalize(scores)
        # 屏蔽掉对角的比对分数
        scores = scores - c
        # 除以温度，可用于放大分数
        scores /= self._temp

        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels, scores))
