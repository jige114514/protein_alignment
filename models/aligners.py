"""Keras Layers for differentiable local sequence alignment."""

from typing import Sequence

import tensorflow as tf

import pairs as pairs_lib
import smith_waterman
from models import initializers


class PairwiseBilinearDense(tf.keras.layers.Layer):
    """Computes (learnable) bilinear form for (batched) sets of vector pairs.
    用编码后的蛋白质序列建模substitution scores, gap open和 gap extend penalties，详见论文S1.2.2
    参数函数采用双线性形式"""

    def __init__(
            self,
            use_kernel=True,
            use_bias=True,
            trainable_kernel=True,
            trainable_bias=True,
            kernel_init=None,
            bias_init='Zeros',
            symmetric_kernel=True,
            dropout=0.0,
            symmetric_dropout=True,
            sqrt_norm=True,
            activation=None,
            mask_penalty=-1e9,
            **kwargs):
        super().__init__(**kwargs)
        if kernel_init is None:
            kernel_init = initializers.SymmetricKernelInitializer()
        bias_init = tf.keras.initializers.get(bias_init)
        self._use_kernel = use_kernel
        self._use_bias = use_bias
        self._trainable_kernel = trainable_kernel
        self._trainable_bias = trainable_bias
        self._kernel_init = kernel_init
        self._bias_init = bias_init
        self._symmetric_kernel = symmetric_kernel
        self._dropout = dropout
        self._symmetric_dropout = symmetric_dropout
        self._sqrt_norm = sqrt_norm
        self._activation = activation
        self._mask_penalty = mask_penalty

    def build(self, input_shape):
        if self._use_kernel:
            emb_dim = input_shape[-1]
            # 权重矩阵
            self.kernel = self.add_weight(
                shape=(emb_dim, emb_dim),
                initializer=self._kernel_init,
                trainable=self._trainable_kernel,
                name='bilinear_form_kernel')
        if self._use_bias:
            # 偏置
            self.bias = self.add_weight(
                shape=(),
                initializer=self._bias_init,
                trainable=self._trainable_bias,
                name='bilinear_form_bias')
        noise_shape = None
        if self._symmetric_dropout:
            noise_shape = [input_shape[0]] + [1] + input_shape[2:]
        self.dropout = tf.keras.layers.Dropout(
            rate=self._dropout, noise_shape=noise_shape)

    def call(self, inputs, mask=None, training=None):
        """Evaluates bilinear form for (batched) sets of vector pairs.

        Args:
          inputs: a tf.Tensor<float>[batch, 2, len, dim] representing two inputs.
          mask: a tf.Tensor<float>[batch, 2, len] to account for padding.
          training: whether to run the layer for train (True), eval (False) or let
            the Keras backend decide (None).

        Returns:
          A tf.Tensor<float>[batch, len, len] s.t.
            out[n][i][j] := activation( (x[n][i]^{T} W y[n][j]) / norm_factor + b),
          where the bilinear form matrix W can optionally be set to be the identity
          matrix (use_kernel = False) or optionally frozen to its initialization
          value (trainable_kernel = False) and the scalar bias b can be optionally
          set to zero (use_bias = False) or likewise optionally frozen to its
          initialization value (trainable_bias=False). If sqrt_norm is True, the
          scalar norm_factor above is set to sqrt(d), following dot-product
          attention. Otherwise, norm_factor = 1.0.
          Finally, if either masks_x[n][i] = 0 or masks_y[n][j] = 0 and mask_penalty
          is not None, then
            out[n][i][j] = mask_penalty
          instead.
        """
        inputs = self.dropout(inputs, training=training)
        x, y = inputs[:, 0], inputs[:, 1]
        if not self._use_kernel:
            output = tf.einsum('ijk,ilk->ijl', x, y)
        else:
            w = self.kernel
            if self._symmetric_kernel:
                w = 0.5 * (w + tf.transpose(w))
            output = tf.einsum('nir,rs,njs->nij', x, w, y)
        if self._sqrt_norm:
            dim_x, dim_y = tf.shape(x)[-1], tf.shape(y)[-1]
            dim = tf.sqrt(tf.cast(dim_x * dim_y, output.dtype))
            output /= tf.sqrt(dim)
        if self._use_bias:
            output += self.bias
        if self._activation is not None:
            output = self._activation(output)
        if self._mask_penalty is not None and mask is not None:
            paired_masks = pairs_lib.pair_masks(mask[:, 0], mask[:, 1])
            output = tf.where(paired_masks, output, self._mask_penalty)

        return output


class ContextualGapPenalties(tf.keras.Model):
    """Wraps untied contextual gap penalty parameters for differentiable SW.

    Gap open and gap extend penalties will be computed without parameter sharing.
    """

    def __init__(self,
                 gap_open_cls=None,
                 gap_extend_cls=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._gap_open = PairwiseBilinearDense(bias_init=tf.keras.initializers.Constant(11.0),
                                               activation=tf.keras.activations.softplus,
                                               mask_penalty=1e9) if gap_open_cls is None else gap_open_cls()
        self._gap_extend = PairwiseBilinearDense(bias_init=tf.keras.initializers.Constant(0.0),
                                                 activation=tf.keras.activations.softplus,
                                                 mask_penalty=1e9) if gap_extend_cls is None else gap_extend_cls()

    def call(self,
             embeddings,
             mask=None,
             training=None):
        """Computes contextual gap open and gap extend params from embeddings.

        Args:
          embeddings: a tf.Tensor<float>[batch, 2, len, dim] representing the
            embeddings of the two inputs.
          mask: a tf.Tensor<float>[batch, 2, len] representing the padding masks of
            the two inputs.
          training: whether to run the layer for train (True), eval (False) or let
            the Keras backend decide (None).

        Returns:
          A 2-tuple (gap_open, gap_extend) of tf.Tensor<float>[batch, len, len].
        """
        return (self._gap_open(embeddings, mask=mask, training=training),
                self._gap_extend(embeddings, mask=mask, training=training))


class SoftAligner(tf.keras.Model):
    """Computes soft Smith-Waterman scores via regularization."""

    def __init__(self,
                 similarity_cls=PairwiseBilinearDense,
                 gap_pen_cls=ContextualGapPenalties,
                 align_fn=smith_waterman.perturbed_alignment_score,
                 eval_align_fn=smith_waterman.unperturbed_alignment_score,
                 trainable=True,
                 **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self._similarity = similarity_cls()
        self._gap_pen = gap_pen_cls()
        self._align_fn = align_fn
        self._eval_align_fn = align_fn if eval_align_fn is None else eval_align_fn

    def call(self, embeddings, mask=None, training=None):
        """Computes soft Smith-Waterman scores via regularization.

        Args:
          embeddings: a tf.Tensor<float>[batch, 2, len, dim] containing pairs of
            sequence embeddings (with the sequence lengths).
            --batch是指两个一组的蛋白质序列组数，len是序列长度，dim是嵌入维度
          mask: An optional token mask to account for padding.
          training: whether to run the layer for train (True), eval (False) or let
            the Keras backend decide (None).

        Returns:
          An AlignmentOutput which is a 3-tuple made of:
            - The alignment scores: tf.Tensor<float>[batch].
            - If provided by the alignment function, the alignment matrix as a
              tf.Tensor<int>[batch, len, len, 9]. Otherwise None.
            - A 3-tuple containing the Smith-Waterman parameters: similarities, gap
              open and gap extend. Similaries is tf.Tensor<float>[batch, len, len],
              the gap penalties can be either tf.Tensor<float>[batch] or
              tf.Tensor<float>[batch, len, len].
        """
        sim_mat = self._similarity(embeddings, mask=mask, training=training)
        gap_open, gap_extend = self._gap_pen(embeddings, mask=mask, training=training)
        sw_params = (sim_mat, gap_open, gap_extend)
        results = (self._align_fn if training else self._eval_align_fn)(*sw_params)
        results = (results,) if not isinstance(results, Sequence) else results
        # TODO(oliviert): maybe inject some metrics here.
        return (results + (None,))[:2] + (sw_params,)
