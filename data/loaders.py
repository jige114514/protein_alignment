"""Builds a dataset with uniref sequences."""

import os

import numpy as np
import tensorflow as tf


class TSVLoader:
    """Creates tf.data.Dataset instances from TSV files with a header."""

    def __init__(self,
                 folder,
                 file_pattern='*.tsv',
                 field_delim='\t',
                 use_quote_delim=False):
        self._folder = folder
        self._file_pattern = file_pattern
        self._field_delim = field_delim
        self._use_quote_delim = use_quote_delim
        self._field_names = None

    def _list_files(self, split):
        pattern = os.path.join(self._folder, split, self._file_pattern)
        return tf.io.gfile.glob(pattern)

    @property
    def field_names(self):
        if self._field_names is None:
            filename = self._list_files('train')[0]
            with tf.io.gfile.GFile(filename, 'r') as f:
                header = f.readline().strip()
            self._field_names = header.split(self._field_delim)
        return self._field_names

    def _csv_dataset_fn(self, filenames):
        return tf.data.experimental.CsvDataset(
            filenames,
            record_defaults=[tf.string] * len(self.field_names),
            header=True,
            field_delim=self._field_delim,
            use_quote_delim=self._use_quote_delim)

    def load(self, split):
        """Creates CSVDataset for split."""
        files = self._list_files(split)
        files_ds = tf.data.Dataset.from_tensor_slices(np.array(files, dtype=str))
        ds = files_ds.interleave(
            self._csv_dataset_fn,
            cycle_length=16,
            block_length=16,
            num_parallel_calls=tf.data.AUTOTUNE)
        return ds.map(
            lambda *ex: {k: v for k, v in zip(self.field_names, ex)},
            num_parallel_calls=tf.data.AUTOTUNE)
