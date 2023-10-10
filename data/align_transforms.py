"""Transformations for pairwise sequence alignment."""

import tensorflow as tf

from data import transforms


class CreateAlignmentTargets(transforms.Transform):
    """Creates targets for pairwise sequence alignment task."""
    # Constants for (integer) encoding of alignment states.
    _GAP_IN_X = -1
    _MATCH = 0
    _GAP_IN_Y = 1
    _START = 2
    # Integer-encoding for special initial transition.
    _INIT_TRANS = 0

    def __init__(self,
                 gap_token='.',
                 n_prepend_tokens=0,
                 **kwargs):
        super().__init__(**kwargs)
        self._gap_token = gap_token
        self._n_prepend_tokens = n_prepend_tokens

        # Transition look-up table (excluding special initial transition).
        look_up = {
            (self._MATCH, self._MATCH): 1,
            (self._GAP_IN_X, self._MATCH): 2,
            (self._GAP_IN_Y, self._MATCH): 3,
            (self._MATCH, self._GAP_IN_X): 4,
            (self._GAP_IN_X, self._GAP_IN_X): 5,
            (self._GAP_IN_Y, self._GAP_IN_X): 9,  # "forbidden" transition.
            (self._MATCH, self._GAP_IN_Y): 6,
            (self._GAP_IN_X, self._GAP_IN_Y): 7,
            (self._GAP_IN_Y, self._GAP_IN_Y): 8,
        }
        # Builds data structures for efficiently encoding transitions.
        self._hash_fn = lambda d0, d1: 3 * (d1 + 1) + (d0 + 1)
        hashes = [self._hash_fn(d0, d1) for (d0, d1) in look_up]
        self._trans_encoder = tf.scatter_nd(indices=[[x] for x in hashes],
                                            updates=list(look_up.values()),
                                            shape=[max(hashes) + 1])
        self._trans_encoder = tf.cast(self._trans_encoder, tf.int32)
        self._init_trans = tf.convert_to_tensor([self._INIT_TRANS], dtype=tf.int32)

    def call(self, seq1, seq2):
        """Creates targets for pairwise sequence alignment task from proj. MSA rows.

        Given a pair of projected rows from an MSA (i.e., with positions at which
        both rows have a gap removed), the ground-truth alignment targets are
        obtained by:
        1) Each position in the projected MSA is classified as _MATCH, _GAP_IN_X or
           _GAP_IN_Y.
        2) The positions of match states are retrieved, as well as the starting
           position of each sequence in the ground-truth (local) alignment.
        3) Positions before the first match state or after the last match state are
           discarded, as these do not belong to the local ground-truth alignment.
        4) For each pair of consecutive match states, where consecutive here is to
           be understood when ignoring non-match states, it is checked whether there
           are BOTH _GAP_IN_X and _GAP_IN_Y states in between.
        5) For each pair of consecutive match states with both _GAP_IN_X and
           _GAP_IN_Y states in between, these states are canonically sorted to
           ensure all _GAP_IN_X states occur first, being followed by all _GAP_IN_Y
           states.
        6) We encode transitions, that is, ordered tuples (s_old, s_new) of states
           using the 9 hidden state model described in `look_up` (c.f. `init`), with
           initial transition (_START, _MATCH) encoded as in `self._init_trans`.
        7) Given the new sequence of states, we reconstructed the positions in each
           sequence where those states would occur.
        8) Finally, optionally, if any special tokens are to be prepended to the
           sequences after this transformation, the ground-truth alignment targets
           will be adjusted accordingly. Note, however, that tokens being appended
           require no further modification.

        Args:
          seq1: A tf.Tensor<int>[len], representing the first proj. row of the MSA.
          seq2: A tf.Tensor<int>[len], representing the second proj. row of the MSA.

        Returns:
          A tf.Tensor<int>[3, tar_len] with three stacked tf.Tensor<int>[tar_len],
          pos1, pos2 and enc_trans, such that (pos1[i], pos2[i], enc_trans[i])
          represents the i-th transition in the ground-truth alignment. For example,
            (pos1[0], pos2[0], enc_trans[0]) = (1, 1, 3)
          would represent that the first transition in the ground-truth alignment is
          from the start state _START to the _MATCH(1,1) state whereas
            (pos1[2], pos2[2], enc_trans[2]) = (2, 5, 4)
          would represent that the third transition in the ground-truth alignment is
          from the match state _MATCH(2, 4) to the gap in X state _GAP_IN_X(2, 5).
          Both pos1 and pos2 use one-based indexing, reserving the use of the value
          zero for padding. In rare cases where the sequence pair has no aligned
          characters, tar_len will be zero.
        """
        keep_indices1 = tf.cast(
            self._vocab.compute_mask(seq1, self._gap_token), tf.int32)
        keep_indices2 = tf.cast(
            self._vocab.compute_mask(seq2, self._gap_token), tf.int32)
        states = keep_indices1 - keep_indices2
        m_states = tf.cast(
            tf.reshape(tf.where(states == self._MATCH), [-1]), tf.int32)
        n_matches = len(m_states)
        if n_matches == 0:
            return tf.zeros([3, 0], tf.int32)
        start, end = m_states[0], m_states[-1]
        offset1 = tf.reduce_sum(keep_indices1[:start])
        offset2 = start - offset1
        offset1 += self._n_prepend_tokens
        offset2 += self._n_prepend_tokens
        states = states[start:end + 1]
        keep_indices1 = keep_indices1[start:end + 1]
        keep_indices2 = keep_indices2[start:end + 1]
        m_states -= start
        segment_ids = tf.cumsum(tf.scatter_nd(
            m_states[1:, tf.newaxis],
            tf.ones(n_matches - 1, dtype=tf.int32),
            shape=[len(states)]))
        aux1 = tf.math.segment_sum(1 - keep_indices1, segment_ids)[:-1]
        aux2 = tf.math.segment_max(1 - keep_indices2, segment_ids)[:-1]
        gap_gap_trans_m_states_indices = tf.reshape(tf.where(aux1 * aux2), [-1])
        if len(gap_gap_trans_m_states_indices) > 0:  # pylint: disable=g-explicit-length-test
            for idx in gap_gap_trans_m_states_indices:
                s_i, e_i = m_states[idx] + 1, m_states[idx + 1]
                m_i = s_i + aux1[idx]
                v_x = tf.fill([aux1[idx]], self._GAP_IN_X)
                v_y = tf.fill([e_i - m_i], self._GAP_IN_Y)
                states = tf.raw_ops.TensorStridedSliceUpdate(
                    input=states, begin=[s_i], end=[m_i], strides=[1], value=v_x)
                states = tf.raw_ops.TensorStridedSliceUpdate(
                    input=states, begin=[m_i], end=[e_i], strides=[1], value=v_y)
        # Builds transitions.
        enc_trans = tf.gather(
            self._trans_encoder, self._hash_fn(states[:-1], states[1:]))
        enc_trans = tf.concat([self._init_trans, enc_trans], 0)
        # Positions such that (pos1[i], pos2[i]) for i = 0, ..., align_len - 1
        # describes the alignment "path".
        pos1 = offset1 + tf.cumsum(tf.cast(states >= self._MATCH, tf.int32))
        pos2 = offset2 + tf.cumsum(tf.cast(states <= self._MATCH, tf.int32))
        return tf.stack([pos1, pos2, enc_trans])
