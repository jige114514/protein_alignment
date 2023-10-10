"""Implements models to embed biological sequences as vector sequences."""

import functools

import tensorflow as tf

import vocabulary
from models import activations
from models import initializers

try:
    # pytype: disable=import-error
    from official.nlp.modeling import layers as nlp_layers  # pylint: disable=g-import-not-at-top
    # pytype: enable=import-error
except Exception:  # pylint: disable=broad-except
    pass


class Encoder(tf.keras.Model):
    """A generic sequence encoder."""

    def __init__(self,
                 vocab=None,
                 mask_special_tokens=False,
                 trainable=True,
                 **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self._vocab = vocabulary.get_default() if vocab is None else vocab
        self._mask_special_tokens = mask_special_tokens

    def compute_mask(self,
                     inputs,
                     mask=None,
                     mask_special_tokens=None):
        """Standard keras method."""
        del mask
        # 序列<EOS>及之前的词元mask设为True，后面全设为False，即padding（0）位置设为False
        mask = self._vocab.padding_mask(inputs)

        # Overrides `self._mask_special_tokens` with `mask_special_tokens` if given.
        if mask_special_tokens is None:
            mask_special_tokens = self._mask_special_tokens
        # 将特殊词元位置设为False
        if mask_special_tokens:
            mask = tf.math.logical_and(mask, self._vocab.special_token_mask(inputs))

        return mask


class TransformerEncoder(Encoder):
    """Encoder with a transformer.
    transformer编码器默认使用6层，多注意头12个，嵌入维度768，多层感知器隐藏层维度3072，激活函数为GELU
    详见论文S1.2.1"""

    def __init__(
            self,
            emb_dim=768,
            num_layers=6,
            num_heads=12,
            mlp_dim=3072,
            mlp_act=activations.approximate_gelu,
            output_dropout=0.1,
            attention_dropout=0.1,
            mlp_dropout=0.1,
            norm_first=True,
            norm_input=False,
            norm_output=True,
            causal=False,
            trainable_posemb=False,
            posemb_init=None,
            aaemb_init=None,
            kernel_init='GlorotUniform',
            aaemb_scale_factor=None,
            max_len=1024,
            **kwargs):
        super().__init__(**kwargs)
        # 初始化位置编码
        if posemb_init is None:
            posemb_init = initializers.HarmonicEmbeddings(scale_factor=1e-4, max_freq=1.0)
        # 初始化氨基酸嵌入
        if aaemb_init is None:
            aaemb_init = tf.initializers.RandomNormal(stddev=1.0)
        kernel_init = tf.keras.initializers.get(kernel_init)
        self._causal = causal
        self.posemb_layer = nlp_layers.PositionEmbedding(
            max_length=max_len,
            initializer=posemb_init,
            trainable=trainable_posemb,
            name='embeddings/positional')
        self.aaemb_layer = nlp_layers.OnDeviceEmbedding(
            vocab_size=len(self._vocab),
            embedding_width=emb_dim,
            initializer=aaemb_init,
            scale_factor=aaemb_scale_factor,
            name='embeddings/aminoacid')
        layer_norm_cls = functools.partial(
            tf.keras.layers.LayerNormalization, axis=-1, epsilon=1e-12)
        self._input_norm_layer = (
            layer_norm_cls(name='embeddings/layer_norm') if norm_input else None)
        self._output_norm_layer = (
            layer_norm_cls(name='output/layer_norm') if norm_output else None)
        self._dropout_layer = tf.keras.layers.Dropout(
            rate=output_dropout, name='embeddings/dropout')
        self._attention_mask = nlp_layers.SelfAttentionMask()
        self._transformer_layers = []
        for i in range(num_layers):
            self._transformer_layers.append(nlp_layers.TransformerEncoderBlock(
                num_attention_heads=num_heads,
                inner_dim=mlp_dim,
                inner_activation=mlp_act,
                output_dropout=output_dropout,
                attention_dropout=attention_dropout,
                inner_dropout=mlp_dropout,
                kernel_initializer=kernel_init,
                norm_first=norm_first,
                name=f'transformer/layer_{i}'))

    def call(self, inputs):
        aa_embeddings = self.aaemb_layer(inputs)
        pos_embeddings = self.posemb_layer(aa_embeddings)
        # 氨基酸嵌入编码加上位置编码
        embeddings = aa_embeddings + pos_embeddings
        # 对输入的嵌入进行LayerNorm（可选，默认不用）
        if self._input_norm_layer is not None:
            embeddings = self._input_norm_layer(embeddings)  # pylint: disable=not-callable
        # 通过dropout层
        embeddings = self._dropout_layer(embeddings)

        mask = self._vocab.padding_mask(inputs)
        attention_mask = self._attention_mask(embeddings, tf.cast(mask, embeddings.dtype))
        if self._causal:
            attention_shape = tf.shape(attention_mask)
            len1, len2 = attention_shape[1], attention_shape[2]
            causal_mask = tf.range(len1)[:, None] >= tf.range(len2)[None, :]
            causal_mask = tf.cast(tf.expand_dims(causal_mask, 0), embeddings.dtype)
            attention_mask *= causal_mask

        # 通过6层transformer编码层
        for layer in self._transformer_layers:
            embeddings = layer((embeddings, attention_mask))

        # 对输出进行LayerNorm（可选，默认使用）
        if self._output_norm_layer is not None:
            embeddings = self._output_norm_layer(embeddings)

        return embeddings
