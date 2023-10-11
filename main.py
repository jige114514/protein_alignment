import random

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import infer, vocabulary, alignment, smith_waterman_np
from models import dedal, encoders, aligners, homology, simcse
from train import learning_rate_schedules, losses, align_metrics
from data import align_transforms, loaders
import pandas as pd


def equal(x, y):
    equal_array = tf.equal(x, y)
    equal_array = tf.cast(equal_array, tf.int32)
    return tf.reduce_sum(equal_array) == tf.reduce_sum(tf.ones_like(equal_array))


df = pd.read_csv('pfam/pfasum60.csv', sep='\t', index_col=0)


def get_substitution_matrix(seq1, seq2):
    len1 = len(seq1)
    len2 = len(seq2)
    sm = np.zeros((len1, len2))
    for i in range(len1):
        for j in range(len2):
            sm[i, j] = df.loc[seq1[i], seq2[j]]
    return sm


def get_data():
    tsvLoader = loaders.TSVLoader('path/to/alignment/data')
    dataset = tsvLoader.load('train')
    data = list(dataset.as_numpy_iterator())
    seqs = set()
    for i, seq_pair in enumerate(data):
        for k, v in seq_pair.items():
            data[i][k] = v.decode('utf-8')
        seqs.add(seq_pair['sequence_x'])
        seqs.add(seq_pair['sequence_y'])
    seqs = list(seqs)
    return data, seqs


data, seqs = get_data()


class Dataset:
    def __init__(self, batch):
        assert batch % 2 == 0
        random.shuffle(seqs)
        self.seqs = [item for seq in seqs for item in [seq] * 2]
        self.cnt = 0
        self.batch = batch
        self.l = len(seqs)

    def __iter__(self):
        return self

    def __next__(self):
        if (self.cnt + 1) * self.batch >= self.l:
            raise StopIteration
        cur_seqs = self.seqs[self.cnt * self.batch:(self.cnt + 1) * self.batch]
        inputs = [infer.preprocess(cur_seqs[i * 2], cur_seqs[i * 2 + 1]) for i in
                  range(self.batch // 2)]
        inputs = tf.concat(inputs, 0)
        self.cnt += 1
        align_values = np.zeros((self.batch, self.batch))
        go = np.ones((1,)) * 15
        ge = np.ones((1,)) * 1.5
        for i in range(self.batch - 1):
            for j in range(i + 1, self.batch):
                sm = get_substitution_matrix(cur_seqs[i], cur_seqs[j])
                sm = np.stack([sm], 0)
                align_values[i][j] = align_values[j][i] = smith_waterman_np.soft_sw_affine(sm, go, ge)
        labels = align_values >= 200
        labels = tf.cast(labels, tf.float16)
        labels = labels / tf.reduce_sum(labels, 1, True)
        return inputs, labels


dedal_model = hub.KerasLayer("dedal_3", trainable=True)
# dedal_model = dedal.DedalLight(
#     encoder=encoders.TransformerEncoder(),
#     aligner=aligners.SoftAligner(),
#     homology_head=homology.LogCorrectedLogits())


def eval(data):
    create_alignment_targets = align_transforms.CreateAlignmentTargets()
    vocab = vocabulary.get_default()
    cnt = 0
    correct = np.zeros(8)
    total = np.zeros(8)
    total2 = np.zeros(8)
    for pair in data:
        inputs = infer.preprocess(pair['sequence_x'], pair['sequence_y'])
        true_alignment = create_alignment_targets.call(
            tf.convert_to_tensor(vocab.encode(pair['gapped_sequence_x']), tf.int32),
            tf.convert_to_tensor(vocab.encode(pair['gapped_sequence_y']), tf.int32))
        true_alignments = tf.stack([true_alignment], 0)
        pred_alignment = dedal_model(inputs, training=False)
        true_paths = alignment.alignments_to_paths(true_alignments, 512, 512)
        match_indices_pred = alignment.paths_to_state_indicators(pred_alignment['paths'], 'match')[0]
        match_indices_true = alignment.paths_to_state_indicators(true_paths, 'match')[0]
        PID = pair["percent_identity"]
        pid_loc = min(7, int(float(PID) / 0.1))
        correct[pid_loc] += tf.reduce_sum(
            tf.cast(tf.logical_and(tf.cast(match_indices_pred, tf.bool), tf.cast(match_indices_true, tf.bool)),
                    tf.float32))
        total[pid_loc] += tf.reduce_sum(match_indices_pred)
        total2[pid_loc] += tf.reduce_sum(match_indices_true)
        precision = correct / total
        recall = correct / total2
        F1 = 2 * (precision * recall) / (precision + recall)

        if cnt > 0 and cnt % 400 == 0:
            print(F1)
        if 0 not in total or cnt > 400:
            break
        cnt += 1


def train(batch):
    simcse_model = simcse.SimCSE(dedal_model)
    criterion = losses.ContrastiveLoss()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedules.InverseSquareRootDecayWithWarmup(lr_max=1e-4,
                                                                               warmup_steps=8_000),
        epsilon=1e-08, clipnorm=1.0)

    for epoch in range(5):
        steps = 0
        for inputs, labels in Dataset(batch):
            steps += 1
            with tf.GradientTape() as g:
                scores = simcse_model(inputs, training=True)
                loss = criterion(scores, labels)
                grad = g.gradient(loss, dedal_model.trainable_variables)
                optimizer.apply_gradients(zip(grad, dedal_model.trainable_variables))
                if steps % 1 == 0:
                    print(loss)
                if steps % 10 == 0:
                    eval(data)


eval(data)
train(8)
eval(data)
