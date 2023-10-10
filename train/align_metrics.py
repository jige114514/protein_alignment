"""Custom metrics for sequence alignment.

This module defines the following types, which serve as inputs to all metrics
implemented here:

+ GroundTruthAlignment is A tf.Tensor<int>[batch, 3, align_len] that can be
  written as tf.stack([pos_x, pos_y, enc_trans], 1) such that
    (pos_x[b][i], pos_y[b][i], enc_trans[b][i]) represents the i-th transition
  in the ground-truth alignment for example b in the minibatch.
  Both pos_x and pos_y are assumed to use one-based indexing and enc_trans
  follows the (categorical) 9-state encoding of edge types used throughout
  `learning/brain/research/combini/diff_opt/alignment/tf_ops.py`.

+ SWParams is a tuple (sim_mat, gap_open, gap_extend) parameterizing the
  Smith-Waterman LP such that
  + sim_mat is a tf.Tensor<float>[batch, len1, len2] (len1 <= len2) with the
    substitution values for pairs of sequences.
  + gap_open is a tf.Tensor<float>[batch, len1, len2] (len1 <= len2) or
    tf.Tensor<float>[batch] with the penalties for opening a gap. Must agree
    in rank with gap_extend.
  + gap_extend is a tf.Tensor<float>[batch, len1, len2] (len1 <= len2) or
    tf.Tensor<float>[batch] with the penalties for extending a gap. Must agree
    in rank with gap_open.

+ AlignmentOutput is a tuple (solution_values, solution_paths, sw_params) such
  that
  + 'solution_values' contains a tf.Tensor<float>[batch] with the (soft) optimal
    Smith-Waterman scores for the batch.
  + 'solution_paths' contains a tf.Tensor<float>[batch, len1, len2, 9] that
    describes the optimal soft alignments.
  + 'sw_params' is a SWParams tuple as described above.
"""

import tensorflow as tf

import alignment


def confusion_matrix(
        alignments_true,
        sol_paths_pred):
    """Computes true, predicted and actual positives for a batch of alignments."""
    batch_size = tf.shape(alignments_true)[0]

    # Computes the number of true positives per example as an (sparse) inner
    # product of two binary tensors of shape (batch_size, len_x, len_y) via
    # indexing. Entirely avoids materializing one of the two tensors explicitly.
    match_indices_true = alignment.alignments_to_state_indices(
        alignments_true, 'match')  # [n_aligned_chars_true, 3]
    match_indicators_pred = alignment.paths_to_state_indicators(
        sol_paths_pred, 'match')  # [batch, len_x, len_y]
    batch_indicators = match_indices_true[:, 0]  # [n_aligned_chars_true]
    matches_flat = tf.gather_nd(
        match_indicators_pred, match_indices_true)  # [n_aligned_chars_true]
    true_positives = tf.math.unsorted_segment_sum(
        matches_flat, batch_indicators, batch_size)  # [batch]

    # Compute number of predicted and ground-truth positives per example.
    pred_positives = tf.reduce_sum(match_indicators_pred, axis=[1, 2])
    # Note(fllinares): tf.math.bincount unsupported in TPU :(
    cond_positives = tf.math.unsorted_segment_sum(
        tf.ones_like(batch_indicators, tf.float32),
        batch_indicators,
        batch_size)  # [batch]
    return true_positives, pred_positives, cond_positives


class AlignmentPrecisionRecall(tf.metrics.Metric):
    """Implements precision and recall metrics for sequence alignment."""

    def __init__(self,
                 name='alignment_pr',
                 threshold=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self._threshold = threshold
        self._true_positives = tf.metrics.Mean()  # TP
        self._pred_positives = tf.metrics.Mean()  # TP + FP
        self._cond_positives = tf.metrics.Mean()  # TP + FN

    def update_state(
            self,
            alignments_true,
            alignments_pred,
            sample_weight=None):
        """Updates TP, TP + FP and TP + FN for a batch of true, pred alignments."""
        if alignments_pred[1] is None:
            return

        sol_paths_pred = alignments_pred[1]
        if self._threshold is not None:  # Otherwise, we assume already binarized.
            sol_paths_pred = tf.cast(sol_paths_pred >= self._threshold, tf.float32)

        true_positives, pred_positives, cond_positives = confusion_matrix(
            alignments_true, sol_paths_pred)

        self._true_positives.update_state(true_positives, sample_weight)
        self._pred_positives.update_state(pred_positives, sample_weight)
        self._cond_positives.update_state(cond_positives, sample_weight)

    def result(self):
        true_positives = self._true_positives.result()
        pred_positives = self._pred_positives.result()
        cond_positives = self._cond_positives.result()
        precision = tf.where(
            true_positives > 0.0, true_positives / pred_positives, 0.0)
        recall = tf.where(
            true_positives > 0.0, true_positives / cond_positives, 0.0)
        f1 = 2.0 * (precision * recall) / (precision + recall)
        return {
            f'{self.name}/precision': precision,
            f'{self.name}/recall': recall,
            f'{self.name}/f1': f1,
        }

    def reset_states(self):
        self._true_positives.reset_states()
        self._pred_positives.reset_states()
        self._cond_positives.reset_states()
