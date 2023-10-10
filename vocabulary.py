"""Classes representing vocabularies (alphabets) over protein strings."""

import itertools
from abc import ABC

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class Vocabulary(ABC):
    """Vocabulary to encode string into integers."""

    MASK = '*'

    def __init__(self,
                 tokens,
                 specials,
                 padding='_',
                 order=None,
                 extra_ids=0):
        """Initializess the vocabulary.

        Args:
          tokens: (Iterable) the main tokens of the vocabulary.
          specials: (Iterable) the special characters, such as EOS, gap, etc.
          padding: the character for padding.
          order: the order between the tokens(0), specials (1) and padding(2).
        """
        self.tokens = tuple(tokens)
        self.specials = tuple(specials)
        self._padding = padding
        self._order = order
        groups = (tokens, specials, [padding])
        order = range(3) if order is None else order
        groups = [groups[i] for i in order]
        self._voc = list(itertools.chain.from_iterable(groups)) + [self.MASK]
        self._indices = {t: i for i, t in enumerate(self._voc)}
        self._extra_ids = extra_ids or 0

    def get_config(self):
        """For keras serialization compatibility."""
        return dict(tokens=self.tokens,
                    specials=self.specials,
                    padding=self._padding,
                    order=self._order)

    def __len__(self):
        return len(self._voc)

    def __contains__(self, token):
        return token in self._indices

    def encode(self, s, skip=None):
        """Returns a list of int-valued tokens representing the input sequence."""
        skip = set() if skip is None else set(skip)
        return [self._indices[c] for c in s if c not in skip]

    def decode(self,
               encoded,
               remove_padding=True,
               remove_specials=True):
        """Returns string representing the input sequence from its int encoding."""
        remove = set(self.specials) if remove_specials else set()
        if remove_padding:
            remove.add(self._padding)
        return ''.join(
            [x for x in [self._voc[i] for i in encoded] if x not in remove])

    @property
    def padding_code(self):
        return self.get(self._padding)

    def get_specials(self, with_padding=True):
        return (self.specials if not with_padding  # pytype: disable=bad-return-type  # trace-all-classes
                else self.specials + (self._padding,))  # pytype: disable=bad-return-type  # trace-all-classes

    @property
    def mask_code(self):
        return self.get(self.MASK)

    def get(self, token, default_value=None):
        """Returns the int encoding of the token if exists or the default value."""
        return self._indices.get(token, default_value)  # pytype: disable=bad-return-type  # trace-all-classes

    def compute_mask(self, inputs, tokens):
        """Computes mask for a batch of input tokens.

        Args:
          inputs: a tf.Tensor of indices.
          tokens: a sequence of strings containing all tokens of interest.

        Returns:
          A binary tf.Tensor of the same size as `inputs`, with value True for
          indices in `inputs` which correspond to no token in `tokens`.
        """
        mask = tf.ones_like(inputs, dtype=tf.bool)
        for token in tokens:
            idx = self._indices[token]
            mask = tf.math.logical_and(mask, inputs != idx)
        return mask

    def padding_mask(self, inputs):
        """Computes padding mask for a batch of input tokens."""
        return self.compute_mask(inputs, [self._padding])

    def special_token_mask(self,
                           inputs,
                           with_mask_token=True):
        """Computes special token mask for a batch of input tokens."""
        tokens = (self.specials + (self.MASK,) if with_mask_token
                  else self.specials)
        return self.compute_mask(inputs, tokens)

    def translate(self, target):
        """Mapping to translate between two instances of Vocabulary.

        Args:
          target: The target Vocabulary instance to be "translated" to.

        Returns:
          A list, mapping the ints of the self vocabulary to the ones of the target.
        """
        return [target.get(token, target.padding_code) for token in self._voc]

    @property
    def _base_vocab_size(self):
        return len(self._voc)

    def _encode(self, s):
        return self.encode(s)

    def _decode(self, ids):
        return self.decode(ids)

    def _encode_tf(self, s):
        return s

    def _decode_tf(self, ids):
        return ids

    @property
    def eos_id(self):
        return self.__len__() + 1

    @property
    def unk_id(self):
        return self.__len__() + 2


@tf.keras.utils.register_keras_serializable()
class SeqIOVocabulary(Vocabulary):
    """Vocabulary compatible with the SeqIO data pipelines."""

    def __init__(  # pylint: disable=super-init-not-called
            self,
            tokens,
            control_specials,
            user_specials,
            extra_ids=95,
    ):
        extra_id_specials = tuple(
            f'�?extra_id_{i}>' for i in reversed(range(1, extra_ids)))
        mask = (self.MASK,)
        self._voc = list(itertools.chain.from_iterable(
            [control_specials, tokens, user_specials, extra_id_specials, mask]))
        self._indices = {t: i for i, t in enumerate(self._voc)}

        self.tokens = tokens
        self.specials = tuple(itertools.chain.from_iterable(
            [control_specials, user_specials, extra_id_specials]))
        self._control_specials = control_specials
        self._user_specials = user_specials
        self._extra_ids = extra_ids
        self._padding = control_specials[0]
        self._order = None

    def get_config(self):
        """For keras serialization compatibility."""
        return dict(tokens=self.tokens,
                    control_specials=self._control_specials,
                    user_specials=self._user_specials,
                    extra_ids=self._extra_ids)


seqio_vocab = SeqIOVocabulary(
    tokens='ARNDCEQGHILKMFPSTWYVBZXJOU',
    control_specials=('_', '>', '?', '<', '▁'),
    user_specials=('.', '-'),
    extra_ids=95,
)


def get_default(vocab=seqio_vocab):
    """A convenient function to gin configure the default vocabulary."""
    return vocab