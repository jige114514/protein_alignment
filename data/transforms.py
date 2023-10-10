"""Transformations to be applied on sequences."""

import abc
from typing import Sequence

import tensorflow as tf

import vocabulary


class Transform(abc.ABC):
    """A generic class for transformations."""

    def __init__(self,
                 on='sequence',
                 out=None,
                 vocab=None):
        self._on = (on,) if isinstance(on, str) else on
        out = self._on if out is None else out
        self._out = (out,) if isinstance(out, str) else out
        self._vocab = vocabulary.get_default() if vocab is None else vocab

    def single_call(self, arg):
        raise NotImplementedError()

    def call(self, *args):
        """Assumes the same order as `on` and `out` for args and outputs.

        This method by default calls the `single_call` method over each argument.
        For Transforms over single argument, the `single_call` method should be
        overwritten. For Transforms over many arguments, one should directly
        overload the `call` method itself.

        Args:
          *args: the argument of the transformation. For Transform over a single
            input, it can be a Sequence of arguments, in which case the Transform
            will be applied over each of them.

        Returns:
          A tf.Tensor or tuple of tf.Tensor.
        """
        result = tuple(self.single_call(arg) for arg in args)
        return result if len(args) > 1 else result[0]

    def __call__(self, inputs):
        keys = set(inputs.keys())
        if not keys.issuperset(self._on):
            raise ValueError(f'The keys of the input ({keys}) are not matching the'
                             f' transform input keys: {self._on}')

        args = tuple(inputs.pop(key) for key in self._on)
        outputs = self.call(*args)
        outputs = (outputs,) if not isinstance(outputs, Sequence) else outputs
        for key, output in zip(self._out, outputs):
            if output is not None:
                inputs[key] = output
        for i, key in enumerate(self._on):
            if key not in self._out:
                inputs[key] = args[i]
        return inputs


class Encode(Transform):
    """
    Encodes a string into a integers.
    将氨基酸和特殊词元映射为整数
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        init = tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(self._vocab._indices.keys())),
            values=tf.constant(list(self._vocab._indices.values()), dtype=tf.int64))
        self._lookup = tf.lookup.StaticVocabularyTable(init, num_oov_buckets=1)

    def single_call(self, arg):
        return tf.cast(self._lookup[tf.strings.bytes_split(arg)], tf.int32)


class CropOrPad(Transform):
    """
    Crops or left/right pads a sequence with the same token.
    在序列后填充0直到达到size
    """

    def __init__(self,
                 size=512,
                 random=True,
                 right=True,
                 token=None,
                 seed=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._size = size
        self._random = random
        self._right = right
        self._token = self._vocab.get(token, self._vocab.padding_code)
        self._seed = seed

    def single_call(self, arg):
        seq_len = tf.shape(arg)[0]
        if seq_len < self._size:
            to_pad = self._size - seq_len
            pattern = [0, to_pad] if self._right else [to_pad, 0]
            arg = tf.pad(arg, [pattern], constant_values=self._token)
        elif seq_len > self._size:
            arg = (
                tf.image.random_crop(arg, [self._size], seed=self._seed)
                if self._random else arg[:self._size])
        arg.set_shape([self._size])
        return arg


class AppendToken(Transform):
    """Left/Right pads a sequence with the a single token."""

    def __init__(self, right=True, token=None, **kwargs):
        super().__init__(**kwargs)
        self._token = self._vocab.get(token, self._vocab.padding_code)
        self._right = right

    def single_call(self, arg):
        pattern = [0, 1] if self._right else [1, 0]
        return tf.pad(arg, [pattern], constant_values=self._token)


class EOS(AppendToken):
    """
    Adds EOS token.
    在序列末尾添加<EOS>（<extra_id_1>的编号126），用以标识序列的末尾
    """

    def __init__(self, token=None, **kwargs):
        """If token is not passed, assumed to be the last special."""
        super().__init__(right=True, token=token, **kwargs)  # Sets vocab and token.
        if token is None:  # Resets self._token to be the last special.
            token = self._vocab.specials[-1]
            self._token = self._vocab.get(token)
