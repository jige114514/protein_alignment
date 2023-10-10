"""Models for alignment tasks."""

import tensorflow as tf

import pairs as pairs_lib


class DedalLight(tf.keras.Model):
    """
    A light-weight model to be easily exported with tf.saved_model.
    轻量级的模型
    """

    def __init__(self, encoder, aligner, homology_head, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.aligner = aligner
        self.homology_head = homology_head

    @tf.function
    def call(self, inputs, training=False, embeddings_only=False):
        embeddings = self.encoder(inputs, training=training)
        if embeddings_only:
            return embeddings

        # Recomputes padding mask, this time ensuring special tokens such as EOS or
        # MASK are zeroed out as well, regardless of the value of the flag
        # `self.encoder._mask_special_tokens`.
        # 将padding和特殊词元的mask设为False
        masks = self.encoder.compute_mask(inputs, mask_special_tokens=True)
        # inputs拆成两个一组的索引
        indices = pairs_lib.consecutive_indices(inputs)
        # inputs拆成两个一组的embedding和mask
        embedding_pairs, mask_pairs = pairs_lib.build(indices, embeddings, masks)
        alignments = self.aligner(embedding_pairs, mask=mask_pairs, training=training)

        # Computes homology scores from SW scores and sequence lengths.
        homology_scores = self.homology_head(alignments, mask=mask_pairs, training=training)
        # Removes "dummy" trailing dimension.
        homology_scores = tf.squeeze(homology_scores, axis=-1)

        return {
            'sw_params': alignments[2],
            'sw_scores': alignments[0],
            'paths': alignments[1],
            'homology_logits': homology_scores,
            'alignments': alignments
        }