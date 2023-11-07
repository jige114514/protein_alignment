import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

import alignment
import infer
import smith_waterman_np
import vocabulary
from data import align_transforms, loaders
from models import contrastive, dedal, encoders, aligners, homology
from parser import args
from train import learning_rate_schedules, losses


def equal(x, y):
    equal_array = tf.equal(x, y)
    equal_array = tf.cast(equal_array, tf.int32)
    return tf.reduce_sum(equal_array) == tf.reduce_sum(tf.ones_like(equal_array))


def setup_seed(seed):
    """
    设置随机种子
    :param seed: 随机种子
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_substitution_matrix(seq1, seq2):
    """
    读取dataframe中的替换矩阵
    :param seq1:
    :param seq2:
    :return:
    """
    len1 = len(seq1)
    len2 = len(seq2)
    sm = np.zeros((len1, len2))
    for i in range(len1):
        for j in range(len2):
            sm[i, j] = df.loc[seq1[i], seq2[j]]
    return sm


def get_data():
    """
    获取数据
    :return: 蛋白质序列对数据信息和所有蛋白质序列
    """
    # 读取蛋白质序列
    tsvLoader = loaders.TSVLoader('path/to/alignment/data')
    dataset = tsvLoader.load('train')
    # data是一个字典，包含成对同源序列的信息
    data = list(dataset.as_numpy_iterator())
    # seqs是所有序列的列表
    seqs = set()
    for i, seq_pair in enumerate(data):
        for k, v in seq_pair.items():
            data[i][k] = v.decode('utf-8')
        seqs.add(seq_pair['sequence_x'])
        seqs.add(seq_pair['sequence_y'])
    seqs = list(seqs)
    return data, seqs


class Dataset:
    """
    有监督对比学习的dataset
    """

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
        # 最后一批若不足batch_size，则舍弃
        if (self.cnt + 1) * self.batch >= self.l:
            raise StopIteration
        cur_seqs = self.seqs[self.cnt * self.batch:(self.cnt + 1) * self.batch]
        # 将成对序列进行预处理，填充到长度512，便于模型处理
        inputs = [infer.preprocess(cur_seqs[i * 2], cur_seqs[i * 2 + 1]) for i in
                  range(self.batch // 2)]
        # 将所有成对序列合并
        inputs = tf.concat(inputs, 0)
        self.cnt += 1
        # 比对分数矩阵
        align_values = np.zeros((self.batch, self.batch))
        # gap_open
        go = np.ones((1,)) * 15
        # gap_extend
        ge = np.ones((1,)) * 1.5
        for i in range(self.batch - 1):
            for j in range(i + 1, self.batch):
                sm = get_substitution_matrix(cur_seqs[i], cur_seqs[j])
                sm = np.stack([sm], 0)
                align_values[i][j] = align_values[j][i] = smith_waterman_np.soft_sw_affine(sm, go, ge)
        # 设置比对分数阈值为200
        # 将在pfasum60使用sw算法得到的比对分数>=200的序列对视为同源序列，作为label
        labels = align_values >= 200
        labels = tf.cast(labels, tf.float16)
        # 将label每一行转换为和为1的概率分布
        labels = labels / tf.reduce_sum(labels, 1, True)
        return inputs, labels


class UnsupervisedDataset:
    """
    无监督simCSE的dataset
    """

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
        # 最后一批若不足batch_size，则舍弃
        if (self.cnt + 1) * self.batch >= self.l:
            raise StopIteration
        cur_seqs = self.seqs[self.cnt * self.batch:(self.cnt + 1) * self.batch]
        # 将成对序列进行预处理，填充到长度512，便于模型处理
        inputs = [infer.preprocess(cur_seqs[i * 2], cur_seqs[i * 2 + 1]) for i in
                  range(self.batch // 2)]
        # 将所有成对序列合并
        inputs = tf.concat(inputs, 0)
        self.cnt += 1
        labels = tf.range(self.batch)
        mask = [1 if i % 2 == 0 else -1 for i in range(self.batch)]
        # 将label从[0,1,2,3,...]转换为[1,0,3,2,...]
        labels += tf.constant(mask)
        return inputs, labels


def eval(data):
    """
    在测试集上评估F1分数
    :param data: 测试集数据
    """
    create_alignment_targets = align_transforms.CreateAlignmentTargets()
    vocab = vocabulary.get_default()
    cnt = 0
    correct = np.zeros(8)
    total = np.zeros(8)
    total2 = np.zeros(8)

    for pair in data:
        inputs = infer.preprocess(pair['sequence_x'], pair['sequence_y'])
        # 序列对的正确比对
        true_alignment = create_alignment_targets.call(
            tf.convert_to_tensor(vocab.encode(pair['gapped_sequence_x']), tf.int32),
            tf.convert_to_tensor(vocab.encode(pair['gapped_sequence_y']), tf.int32))
        true_alignments = tf.stack([true_alignment], 0)
        # 通过模型得到的预测比对
        pred_alignment = dedal_model(inputs, training=False)
        true_paths = alignment.alignments_to_paths(true_alignments, 512, 512)
        match_indices_pred = alignment.paths_to_state_indicators(pred_alignment['paths'], 'match')[0]
        match_indices_true = alignment.paths_to_state_indicators(true_paths, 'match')[0]
        PID = pair["percent_identity"]
        # pid范围0~0.7，每0.1划分一个间隔
        pid_loc = min(7, int(float(PID) / 0.1))
        correct[pid_loc] += tf.reduce_sum(
            tf.cast(tf.logical_and(tf.cast(match_indices_pred, tf.bool), tf.cast(match_indices_true, tf.bool)),
                    tf.float32))
        total[pid_loc] += tf.reduce_sum(match_indices_pred)
        total2[pid_loc] += tf.reduce_sum(match_indices_true)
        precision = correct / total
        recall = correct / total2
        # 计算F1分数
        F1 = 2 * (precision * recall) / (precision + recall)

        if cnt % 100 == 0:
            print(f'F1 = {F1}')
        cnt += 1


def train(batch, epochs):
    """
    对比学习微调
    :param batch: 一批中的序列数（包括复制的序列）
    :param epochs: 训练轮数
    """
    contrastive_model = contrastive.Contrastive(dedal_model)
    criterion = losses.ContrastiveLoss()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedules.InverseSquareRootDecayWithWarmup(lr_max=1e-4,
                                                                               warmup_steps=8_000),
        epsilon=1e-08, clipnorm=1.0)

    # 选择有监督或无监督训练的dataset
    if args.dataset == 'supervised':
        dataset = Dataset(batch)
    else:
        dataset = UnsupervisedDataset(batch)

    # 打印train loss的间隔
    interval = args.print_interval

    for epoch in range(epochs):
        steps = 0
        for inputs, labels in dataset:
            steps += 1
            with tf.GradientTape() as g:
                scores = contrastive_model(inputs, training=True)
                # 将比对分数视为成对序列的相似度
                loss = criterion(scores, labels)
                grad = g.gradient(loss, dedal_model.trainable_variables)
                optimizer.apply_gradients(zip(grad, dedal_model.trainable_variables))
                if steps % interval == 0:
                    print(f'loss = {loss}')


if __name__ == '__main__':
    # 设置随机种子
    setup_seed(args.seed)

    # 读取pfasum60替换矩阵
    df = pd.read_csv('pfam/pfasum60.csv', sep='\t', index_col=0)
    # 获取序列对数据字典和所有蛋白质序列
    data, seqs = get_data()
    # 随机打乱数据
    random.shuffle(data)
    # 前10%数据作为测试集
    test_num = int(len(data) * 0.1)
    test_data = data[:test_num]

    # 载入已训练好的dedal模型
    dedal_model = hub.KerasLayer("dedal_3", trainable=True)
    # dedal_model = dedal.DedalLight(
    #     encoder=encoders.TransformerEncoder(),
    #     aligner=aligners.SoftAligner(),
    #     homology_head=homology.LogCorrectedLogits())  # for debug

    # 先进行初次评估
    eval(test_data)
    # 对比学习微调
    train(args.batch_size, args.epoch)
    # 微调后评估
    eval(test_data)
