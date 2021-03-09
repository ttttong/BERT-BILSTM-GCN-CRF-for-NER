#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:xiongxin
Adjust code for BiLSTM plus CRF based on zhoukaiyin.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import os
from bert import modeling
from bert import optimization
from bert import tokenization
from bilstm_crf.bilstm_crf_layer import BiLSTM_CRF
from bilstm_crf.bilstm_gcn5 import BiLSTM_GCN_CRF3   # gcn输入到bilstm
from bilstm_crf.bilstm_gcn3 import BiLSTM_GCN_CRF     # bilstm输入到gcn
from bilstm_crf.bilstm_gcn4 import BiLSTM_GCN_CRF2   # bilstm+gcn
from metrics import tf_metrics
from metrics import conlleval
import tensorflow as tf
import pickle
import logging
import aux1
import random

# ML的模型中有大量需要tuning的超参数
# flags可以帮助我们通过命令行来动态的更改代码中的参数
# 类似于 argparse
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", None,
                    "The dataset, refer to data_dir")

flags.DEFINE_string("bert_config_file", None,
                    "The config json file corresponding to the pre-trained BERT model.")

flags.DEFINE_string("task_name", "NER",
                    "The name of the task to train.")

flags.DEFINE_string("output_dir", None,
                    "The output directory where the model checkpoints will be written.")

# Other parameters
flags.DEFINE_string("init_checkpoint", None,
                    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_lower_case", True,
                  "Whether to lower case the input text. Should be True for uncased.")

flags.DEFINE_integer("max_seq_length", 128,
                     "The maximum total input sequence length after WordPiece tokenization.")

flags.DEFINE_bool("do_train", True,
                  "Whether to run training.")

flags.DEFINE_bool("do_eval", True,
                  "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_test", True,
                  "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32,
                     "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8,
                     "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("dropout_rate", None,
                   "The rate to dropout cells in embedding layer.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float("warmup_proportion", 0.1,
                   "Proportion of training to perform linear learning rate warmup for. "
                   "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1583,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

flags.DEFINE_bool("use_tpu", False,
                  "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None,
                    "[Optional] TensorFlow master URL.")

flags.DEFINE_integer("num_tpu_cores", 8,
                     "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("bilstm", True,
                  "use bilstm.")

flags.DEFINE_bool("crf", True,
                  "use crf.")

flags.DEFINE_bool("use_pos", True,
                  "use pos feature.")

flags.DEFINE_integer("gcn", 0,  # 0-不用/1-lstm后/2-替代cnn
                     "use gcn.")

# lstm params
flags.DEFINE_integer('lstm_size', 128,
                     'size of lstm units')

flags.DEFINE_integer('num_layers', 1,
                     'number of rnn layers, default is 1')

flags.DEFINE_string('cell', 'lstm',
                    'which rnn cell used')


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, text, label=None, pos=None):
        """
        construct a input example
        :param guid: unique id for the example
        :param text: The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        :param label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.pos = pos


class InputFeatures(object):
    """
    A single set of features of data.
    """

    def __init__(self, ntokens, input_ids, input_mask, segment_ids, label_ids, pos_ids, forward, backward, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.pos_ids = pos_ids
        self.is_real_example = is_real_example
        self.forward = forward
        self.backward = backward
        self.ntokens = ntokens


class DataProcessor(object):
    """
    Base class for data converters for sequence classification data sets.
    """

    def get_train_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the train set.
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the dev set.
        """
        raise NotImplementedError()

    def get_labels(self):
        """
        Gets the list of labels for this data set.
        """
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, set_type):
        """
        customized to read train/dev/test data here!
        """
        pos2id = dict()
        with open('./data/{}/pos2id.pkl'.format(FLAGS.dataset), 'rb') as rf:
            pos2id = pickle.load(rf)

        with open('./data/{}/{}.txt'.format(FLAGS.dataset, set_type), 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            poses = []
            count = 0
            for line in f:
                contents = line.strip()
                if contents.startswith("-DOCSTART-"):
                    words.append('')
                    count += 1
                    continue
                if len(contents) == 0:
                    s = ' '.join([str(pos) for pos in poses])
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([w, l, s])
                    words = []
                    labels = []
                    poses = []
                    count += 1
                    continue
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[1]
                pos = pos2id[line.strip().split(' ')[-1]]
                words.append(word)
                labels.append(label)
                poses.append(pos)
                count += 1
            return lines


class NerProcessor(DataProcessor):
    def _create_example(self, lines, set_type):
        example = []
        for (i, line) in enumerate(lines):
            guid = '{}-{}'.format(set_type, i)
            text = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            pos = tokenization.convert_to_unicode((line[2]))
            example.append(InputExample(guid=guid, text=text, label=label, pos=pos))
        return example

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data('train'), 'train'
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data('dev'), 'dev'
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data('test'), "test"
        )

    def get_labels(self):
        """
        based on ChinaDaily corpus
        'X' is used to represent "##eer","##soo" and char not in vocab!
        """
        return ["X", "B-剧种", "I-剧种", "B-人名", "I-人名", "B-剧目", "I-剧目", "B-乐器", "I-乐器", "B-地点", "I-地点", "B-唱腔曲牌", "I-唱腔曲牌", "B-脚色行当", "I-脚色行当", "O", "[CLS]", "[SEP]"]


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, forward, backward, mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature
    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [B,I,X,O,O,O,X]
    """
    label_map = dict()
    # here start with zero which means that "[PAD]" is zero
    # start with 1, 0 for paddding
    for i, label in enumerate(label_list, 1):
        label_map[label] = i

    with open('./data/{}/label2id.pkl'.format(FLAGS.dataset), 'wb+') as wf:
        pickle.dump(label_map, wf)

    pos2id = dict()
    with open('./data/{}/pos2id.pkl'.format(FLAGS.dataset), 'rb') as rf:
        pos2id = pickle.load(rf)

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    # featurelist = [list(map(int, features.split('-'))) for features in example.feature.split(' ')]
    poslist = example.pos.split(' ')
    tokens = []
    labels = []
    poses = []
    for i, (word, label, pos) in enumerate(zip(textlist, labellist, poslist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for m in range(len(token)):
            if m == 0:
                labels.append(label)
                poses.append(pos)
            else:
                labels.append('X')
                poses.append(0)
    # only Account for [CLS] with "- 1".
    # account for ending signal [SEP], with "- 2"
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0: max_seq_length - 2]
        labels = labels[0: max_seq_length - 2]
        poses = poses[0: max_seq_length - 2]
    ntokens = []
    segment_ids = []
    label_ids = []
    pos_ids = []
    # begin signal [CLS]
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    pos_ids.append(pos2id["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
        pos_ids.append(int(poses[i]))
    # ending signal [SEP]
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])
    pos_ids.append(pos2id["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # use zero to padding sequence
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        pos_ids.append(0)
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(pos_ids) == max_seq_length

    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % example.guid)
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        logging.info("pos_ids: %s" % " ".join([str(x) for x in pos_ids]))
        logging.info("ntokens: %s" % " ".join([str(x) for x in ntokens]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        pos_ids=pos_ids,
        forward=forward,
        backward=backward,
        ntokens=ntokens
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature, ntokens, label_ids


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, forward, backward, mode=None):
    writer = tf.python_io.TFRecordWriter(path=output_file)
    # writer = tf.io.TFRecordWriter(path=output_file)
    batch_tokens = []
    batch_labels = []
    for ex_index, (example, a_fw, a_bw) in enumerate(zip(examples, forward, backward)):
        if ex_index % 5000 == 0:
            logging.info('Writing example {} of {}'.format(ex_index, len(examples)))
        feature, ntokens, label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,
                                                             a_fw, a_bw, mode)
        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=values.flatten()))
            return f

        def create_byte_feature(values):
            f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8') if type(value)==str else value for value in values]))
            return f

        features = collections.OrderedDict()

        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["pos_ids"] = create_int_feature(feature.pos_ids)
        features["forward"] = create_float_feature(feature.forward)
        features["backward"] = create_float_feature(feature.backward)
        features["ntokens"] = create_byte_feature(feature.ntokens)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()
    return batch_tokens, batch_labels


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "pos_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "forward": tf.VarLenFeature(tf.float32),
        "backward": tf.VarLenFeature(tf.float32),
        "ntokens": tf.FixedLenFeature([seq_length], tf.string),
    }

    def _deocde_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            if t.dtype == tf.float32:
                t = tf.sparse_tensor_to_dense(t)
                t = tf.reshape(t, [seq_length, seq_length])
                # print(t)
                # sess = tf.Session()
                # print(sess.run(t))
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _deocde_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def hidden2tag(hiddenlayer, numclass):
    # tf.keras.layers.Dense 封装了output = activation(tf.matmul(input, kernel) + bias)
    # 相当于全连接层的线性变换
    linear = tf.keras.layers.Dense(numclass, activation=None)
    return linear(hiddenlayer)


def softmax_layer(logits, labels, num_labels, mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask, dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12  # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, label_ids, pos_ids, num_labels, forward, backward, use_one_hot_embeddings, ntokens):
    """
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param segment_ids:
    :param label_ids: 标签的idx 表示
    :param pos_ids: 额外特征的idx表示 [batch_size, seq_length]
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :return:
    """
    # 使用数据加载BertModel,获取对应的字embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    # [batch_size, seq_length, embedding_size]
    # use model.get_sequence_output() to get token-level output
    output_embeddings = model.get_sequence_output()
    # embeddings = output_embeddings  # 不用word2vec要把他注释取消

    # word2vec
    from gensim.models import Word2Vec
    model2 = Word2Vec.load('word2vec/wiki.zh.text.model')
    w2v_embeddings = []
    with tf.Session() as sess:
        for token in ntokens.eval(Session=sess):
            if token == '[CLS]' or token == '[SEP]' or token == '**NULL**':
                zeroList = [0 for i in range(100)]
                w2v_embeddings.append(zeroList)
            else:
                i = model2[token]
                w2v_embeddings.append(i)
    # if ntokens == '[CLS]' or ntokens == '[SEP]' or ntokens == '**NULL**':
    #     zeroList = [0 for i in range(100)]
    #     w2v_embeddings.append(zeroList)
    # else:
    #     i = model2[ntokens]
    #     w2v_embeddings.append(i)
    w2v_embeddings = tf.convert_to_tensor(w2v_embeddings)
    embeddings = tf.concat([output_embeddings, w2v_embeddings], -1)

    if FLAGS.use_pos:
        pos2id = {}

        # pos = pos/mark
        with open('./data/{}/pos2id.pkl'.format(FLAGS.dataset), 'rb') as rf:
            pos2id = pickle.load(rf)
        num_pos = 100
        pos_embeddings = tf.one_hot(pos_ids, num_pos, axis=-1)  # 维度 pos_ids*num_pos
        embeddings = tf.concat([embeddings, pos_embeddings], -1)  # 加长embedding向量
        print(embeddings.shape)

    if FLAGS.bilstm or FLAGS.crf:
        '''
        used = tf.sign(tf.abs(input_ids))
        lengths = tf.reduce_sum(used, reduction_indices=1)
        '''
        # [batch_size] 大小的向量，包含了当前batch中的序列长度
        lengths = tf.reduce_sum(input_mask, axis=1)
        max_seq_length = embeddings.shape[1].value

        if FLAGS.gcn == 0:
            bilstm_crf = BiLSTM_GCN_CRF3(embedded_chars=embeddings, lstm_size=FLAGS.lstm_size, cell_type=FLAGS.cell,
                                num_layers=FLAGS.num_layers,
                                dropout_rate=FLAGS.dropout_rate, num_labels=num_labels,
                                max_seq_length=max_seq_length,
                                labels=label_ids,
                                lengths=lengths, is_training=is_training, bilstm=FLAGS.bilstm, crf=FLAGS.crf, forward=forward, backward=backward)
            loss, predict = bilstm_crf.add_bilstm_crf_layer()
        elif FLAGS.gcn == 1:
            bilstm_crf = BiLSTM_GCN_CRF(embedded_chars=embeddings, lstm_size=FLAGS.lstm_size, cell_type=FLAGS.cell,
                                num_layers=FLAGS.num_layers,
                                dropout_rate=FLAGS.dropout_rate, num_labels=num_labels,
                                max_seq_length=max_seq_length,
                                labels=label_ids,
                                lengths=lengths, is_training=is_training, bilstm=FLAGS.bilstm, crf=FLAGS.crf, forward=forward, backward=backward)
            loss, predict = bilstm_crf.add_bilstm_crf_layer()
        elif FLAGS.gcn == 2:
            bilstm_crf = BiLSTM_GCN_CRF2(embedded_chars=embeddings, lstm_size=FLAGS.lstm_size, cell_type=FLAGS.cell,
                                num_layers=FLAGS.num_layers,
                                dropout_rate=FLAGS.dropout_rate, num_labels=num_labels,
                                max_seq_length=max_seq_length,
                                labels=label_ids,
                                lengths=lengths, is_training=is_training, bilstm=FLAGS.bilstm, crf=FLAGS.crf, forward=forward, backward=backward)
            loss, predict = bilstm_crf.add_bilstm_crf_layer(pos_ids)
    else:
        if is_training:
            embeddings = tf.nn.dropout(embeddings, keep_prob=0.9)
        logits = hidden2tag(embeddings, num_labels)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        loss, predict = softmax_layer(logits, label_ids, num_labels, input_mask)

    return (loss, predict)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = {}, shape = {}".format(name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        pos_ids = features["pos_ids"]
        forward = features["forward"]
        backward = features["backward"]
        ntokens = features["ntokens"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, pred_ids) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            label_ids, pos_ids, num_labels, forward, backward, use_one_hot_embeddings, ntokens)

        vars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                vars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        logging.info("**** Trainable Variables ****")

        for var in vars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("  name = {}, shape = {} {}".format(var.name, var.shape, init_string))

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(label_ids, pred_ids):
                try:
                    # confusion matrix
                    cm = tf_metrics.streaming_confusion_matrix(label_ids, pred_ids, num_labels, weights=input_mask)
                    return {
                        "confusion_matrix": cm
                    }
                except Exception as e:
                    logging.error(str(e))

            eval_metrics = (metric_fn, [label_ids, pred_ids])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn
            )
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=pred_ids, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main(_):
    # tf.logging.set_verbosity(tf.logging.INFO)

    for name, value in FLAGS.__flags.items():
        logging.info('{} = {}'.format(name, value.value))

    processors = {
        "ner": NerProcessor
    }

    data_dir = './data/{}/'.format(FLAGS.dataset)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model, was only trained up to sequence length {}".format(
                FLAGS.max_seq_length, bert_config.max_position_embeddings))

    # tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: {}".format(task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case
    )
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host
        )
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法,
    # 并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程;
    # tf 新的架构方法，通过定义model_fn 函数，定义模型,
    # 然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)


    if FLAGS.do_train:
        # 图模型
        a = data_dir+'train.txt'
        sentences = aux1.get_all_sentences(a)
        data = aux1.get_data_from_sentences(sentences)

        forward = []
        backward = []
        for item in data:
            words = item[0]
            sentence = aux1.create_full_sentence(words)
            A_fw, A_bw = aux1.create_graph_from_sentence_and_word_vectors(sentence, FLAGS.max_seq_length)
            forward.append(A_fw)
            backward.append(A_bw)

        # 1. 将数据转化为tf_record 数据
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        print('***train_file***', train_file)
        print('-----------', len(forward))
        _, _ = file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, forward, backward)
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)

        # 2.读取record 数据，组成batch
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)


        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        # 图模型
        a = data_dir+'dev.txt'
        sentences = aux1.get_all_sentences(a)
        data = aux1.get_data_from_sentences(sentences)
        # buckets = aux1.bin_data_into_buckets(data, FLAGS.train_batch_size)
        # random_buckets = sorted(buckets, key=lambda x: random.random())
        forward = []
        backward = []
        for item in data:
            words = item[0]
            sentence = aux1.create_full_sentence(words)
            A_fw, A_bw = aux1.create_graph_from_sentence_and_word_vectors(sentence, FLAGS.max_seq_length)
            forward.append(A_fw)
            backward.append(A_bw)
        eval_examples = processor.get_dev_examples(data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        _, _ = file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, forward, backward)

        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_steps = None
        eval_drop_remainder = True if FLAGS.use_tpu else False
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        output_eval_cm = os.path.join(FLAGS.output_dir, "eval_results_cm.txt")

        with open(output_eval_file, "a+", encoding='utf-8') as writer:
            logging.info("***** Eval results *****")
            writer.write('data:\t{}\n'.format(FLAGS.dataset))
            writer.write('model:\t{}\n'.format(FLAGS.init_checkpoint))
            writer.write('loss:\t{}\n'.format(result['loss']))
            writer.write('global_step:\t{}\n'.format(result['global_step']))
            confusion_matrix = result.get("confusion_matrix", None)
            with open(output_eval_cm, "a+", encoding='utf-8') as fw_cm:
                fw_cm.write('data:\t{}\n'.format(FLAGS.dataset))
                fw_cm.write('model:\t{}\n'.format(FLAGS.init_checkpoint))
                for row in confusion_matrix:
                    for col in row:
                        fw_cm.write('{:1f}\t'.format(col))
                    fw_cm.write('\n')
            try:
                precisions, recalls, fs, acc, kappa = tf_metrics.calculate(confusion_matrix)
                writer.write('Precision: {}\n'.format('\t'.join([str(p) for p in precisions])))
                writer.write('Recall: {}\n'.format('\t'.join([str(r) for r in recalls])))
                writer.write('F1: {}\n'.format('\t'.join([str(f) for f in fs])))
                writer.write('Acc: {}\n'.format(acc))
                writer.write('Kappa: {}\n'.format(kappa))
            except Exception as e:
                logging.error(str(e))

    if FLAGS.do_test:
        with open('./data/{}/label2id.pkl'.format(FLAGS.dataset), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        predict_examples = processor.get_test_examples(data_dir)
        # 图模型
        a = data_dir+'test.txt'
        sentences = aux1.get_all_sentences(a)
        data = aux1.get_data_from_sentences(sentences)
        # buckets = aux1.bin_data_into_buckets(data, FLAGS.train_batch_size)
        # random_buckets = sorted(buckets, key=lambda x: random.random())
        forward = []
        backward = []
        # for bucket in buckets:
        for item in data:
            words = item[0]
            sentence = aux1.create_full_sentence(words)
            A_fw, A_bw = aux1.create_graph_from_sentence_and_word_vectors(sentence, FLAGS.max_seq_length)
            forward.append(A_fw)
            backward.append(A_bw)

        predict_file = os.path.join(FLAGS.output_dir, "test.tf_record")
        batch_tokens, batch_labels = file_based_convert_examples_to_features(
            predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file, forward, backward, mode="test")

        logging.info("***** Running Test*****")
        logging.info("  Num examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_test_file = os.path.join(FLAGS.output_dir, "test_label.txt")
        result_test_file = os.path.join(FLAGS.output_dir, "test_result.txt")

        # here if the tag is "X" means it belong to its before token for convenient evaluate use
        def Writer(output_test_file, result, batch_tokens, batch_labels, id2label):
            with open(output_test_file, 'w+', encoding='UTF-8') as wf:

                if FLAGS.bilstm:
                    predictions = []
                    for _, pred in enumerate(result):
                        predictions.extend(pred)
                else:
                    predictions = result

                for i, prediction in enumerate(predictions):
                    token = batch_tokens[i]
                    if prediction == 0:
                        continue
                    predict = id2label[prediction]
                    true_label = id2label[batch_labels[i]]
                    if token in ['[CLS]', '[SEP]']:
                        continue
                    line = "{}\t{}\t{}\n".format(token, true_label, predict)
                    wf.write(line)

        Writer(output_test_file, result, batch_tokens, batch_labels, id2label)

        test_report = conlleval.return_report(output_test_file)
        logging.info(''.join(test_report))
        with open(result_test_file, 'a+', encoding='UTF-8') as wf:
            wf.write('data:\t{}\n'.format(FLAGS.dataset))
            wf.write('model:\t{}\n'.format(FLAGS.init_checkpoint))
            wf.write(''.join(test_report))


if __name__ == '__main__':
    flags.mark_flag_as_required("dataset")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    if not os.path.exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    LOG_SETTINGS = {
        'format': '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
    }
    FILE_NAME = os.path.join(FLAGS.output_dir, '{}.log'.format('train' if FLAGS.do_train else 'test'))
    logging.basicConfig(
        handlers=[logging.FileHandler(FILE_NAME, encoding="utf-8", mode='a')],
        level=logging.INFO,
        **LOG_SETTINGS
    )
    tf.app.run()
    # tf.compat.v1.app.run()






# #! usr/bin/env python3
# # -*- coding:utf-8 -*-
# """
# Copyright 2018 The Google AI Language Team Authors.
# BASED ON Google_BERT.
# @Author:xiongxin
# Adjust code for BiLSTM plus CRF based on zhoukaiyin.
# """
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import numpy as np
# import collections
# import os
# from bert import modeling
# from bert import optimization
# from bert import tokenization
# from bilstm_crf.bilstm_crf_layer import BiLSTM_CRF
# from bilstm_crf.bilstm_gcn5 import BiLSTM_GCN_CRF3   # gcn输入到bilstm
# from bilstm_crf.bilstm_gcn3 import BiLSTM_GCN_CRF     # bilstm输入到gcn
# from bilstm_crf.bilstm_gcn4 import BiLSTM_GCN_CRF2   # bilstm+gcn
# from metrics import tf_metrics
# from metrics import conlleval
# import tensorflow as tf
# import pickle
# import logging
# import aux1
# import random
#
# # ML的模型中有大量需要tuning的超参数
# # flags可以帮助我们通过命令行来动态的更改代码中的参数
# # 类似于 argparse
# flags = tf.flags
# FLAGS = flags.FLAGS
#
# flags.DEFINE_string("dataset", None,
#                     "The dataset, refer to data_dir")
#
# flags.DEFINE_string("bert_config_file", None,
#                     "The config json file corresponding to the pre-trained BERT model.")
#
# flags.DEFINE_string("task_name", "NER",
#                     "The name of the task to train.")
#
# flags.DEFINE_string("output_dir", None,
#                     "The output directory where the model checkpoints will be written.")
#
# # Other parameters
# flags.DEFINE_string("init_checkpoint", None,
#                     "Initial checkpoint (usually from a pre-trained BERT model).")
#
# flags.DEFINE_bool("do_lower_case", True,
#                   "Whether to lower case the input text. Should be True for uncased.")
#
# flags.DEFINE_integer("max_seq_length", 128,
#                      "The maximum total input sequence length after WordPiece tokenization.")
#
# flags.DEFINE_bool("do_train", True,
#                   "Whether to run training.")
#
# flags.DEFINE_bool("do_eval", True,
#                   "Whether to run eval on the dev set.")
#
# flags.DEFINE_bool("do_test", True,
#                   "Whether to run the model in inference mode on the test set.")
#
# flags.DEFINE_integer("train_batch_size", 32,
#                      "Total batch size for training.")
#
# flags.DEFINE_integer("eval_batch_size", 8,
#                      "Total batch size for eval.")
#
# flags.DEFINE_integer("predict_batch_size", 8,
#                      "Total batch size for predict.")
#
# flags.DEFINE_float("learning_rate", 5e-5,
#                    "The initial learning rate for Adam.")
#
# flags.DEFINE_float("dropout_rate", None,
#                    "The rate to dropout cells in embedding layer.")
#
# flags.DEFINE_float("num_train_epochs", 3.0,
#                    "Total number of training epochs to perform.")
#
# flags.DEFINE_float("warmup_proportion", 0.1,
#                    "Proportion of training to perform linear learning rate warmup for. "
#                    "E.g., 0.1 = 10% of training.")
#
# flags.DEFINE_integer("save_checkpoints_steps", 1583,
#                      "How often to save the model checkpoint.")
#
# flags.DEFINE_integer("iterations_per_loop", 1000,
#                      "How many steps to make in each estimator call.")
#
# flags.DEFINE_string("vocab_file", None,
#                     "The vocabulary file that the BERT model was trained on.")
#
# # import os
# #
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#
# flags.DEFINE_bool("use_tpu", False,
#                   "Whether to use TPU or GPU/CPU.")
#
# flags.DEFINE_string(
#     "tpu_name", None,
#     "The Cloud TPU to use for training. This should be either the name "
#     "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
#     "url.")
#
# flags.DEFINE_string(
#     "tpu_zone", None,
#     "[Optional] GCE zone where the Cloud TPU is located in. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")
#
# flags.DEFINE_string(
#     "gcp_project", None,
#     "[Optional] Project name for the Cloud TPU-enabled project. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")
#
# flags.DEFINE_string("master", None,
#                     "[Optional] TensorFlow master URL.")
#
# flags.DEFINE_integer("num_tpu_cores", 8,
#                      "Only used if `use_tpu` is True. Total number of TPU cores to use.")
#
# flags.DEFINE_bool("bilstm", True,
#                   "use bilstm.")
#
# flags.DEFINE_bool("crf", True,
#                   "use crf.")
#
# flags.DEFINE_bool("use_pos", True,
#                   "use pos feature.")
#
# flags.DEFINE_integer("gcn", 0,
#                      "use gcn.")
#
# # lstm params
# flags.DEFINE_integer('lstm_size', 128,
#                      'size of lstm units')
#
# flags.DEFINE_integer('num_layers', 1,
#                      'number of rnn layers, default is 1')
#
# flags.DEFINE_string('cell', 'lstm',
#                     'which rnn cell used')
#
#
# class InputExample(object):
#     """
#     A single training/test example for simple sequence classification.
#     """
#
#     def __init__(self, guid, text, label=None, pos=None):
#         """
#         construct a input example
#         :param guid: unique id for the example
#         :param text: The untokenized text of the first sequence. For single
#             sequence tasks, only this sequence must be specified.
#         :param label: (Optional) string. The label of the example. This should be
#             specified for train and dev examples, but not for test examples.
#         """
#         self.guid = guid
#         self.text = text
#         self.label = label
#         self.pos = pos
#
#
# class InputFeatures(object):
#     """
#     A single set of features of data.
#     """
#
#     def __init__(self, input_ids, input_mask, segment_ids, label_ids, pos_ids, forward, backward, is_real_example=True):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.label_ids = label_ids
#         self.pos_ids = pos_ids
#         self.is_real_example = is_real_example
#         self.forward = forward
#         self.backward = backward
#
#
# class DataProcessor(object):
#     """
#     Base class for data converters for sequence classification data sets.
#     """
#
#     def get_train_examples(self, data_dir):
#         """
#         Gets a collection of `InputExample`s for the train set.
#         """
#         raise NotImplementedError()
#
#     def get_dev_examples(self, data_dir):
#         """
#         Gets a collection of `InputExample`s for the dev set.
#         """
#         raise NotImplementedError()
#
#     def get_labels(self):
#         """
#         Gets the list of labels for this data set.
#         """
#         raise NotImplementedError()
#
#     @classmethod
#     def _read_data(cls, set_type):
#         """
#         customized to read train/dev/test data here!
#         """
#         pos2id = dict()
#         with open('./data/{}/pos2id.pkl'.format(FLAGS.dataset), 'rb') as rf:
#             pos2id = pickle.load(rf)
#
#         with open('./data/{}/{}.txt'.format(FLAGS.dataset, set_type), 'r', encoding='utf-8') as f:
#             lines = []
#             words = []
#             labels = []
#             poses = []
#             count = 0
#             for line in f:
#                 contents = line.strip()
#                 if contents.startswith("-DOCSTART-"):
#                     words.append('')
#                     count += 1
#                     continue
#                 if len(contents) == 0:
#                     s = ' '.join([str(pos) for pos in poses])
#                     l = ' '.join([label for label in labels if len(label) > 0])
#                     w = ' '.join([word for word in words if len(word) > 0])
#                     lines.append([w, l, s])
#                     words = []
#                     labels = []
#                     poses = []
#                     count += 1
#                     continue
#                 word = line.strip().split(' ')[0]
#                 label = line.strip().split(' ')[1]
#                 pos = pos2id[line.strip().split(' ')[-1]]
#                 words.append(word)
#                 labels.append(label)
#                 poses.append(pos)
#                 count += 1
#             return lines
#
#
# class NerProcessor(DataProcessor):
#     def _create_example(self, lines, set_type):
#         example = []
#         for (i, line) in enumerate(lines):
#             guid = '{}-{}'.format(set_type, i)
#             text = tokenization.convert_to_unicode(line[0])
#             label = tokenization.convert_to_unicode(line[1])
#             pos = tokenization.convert_to_unicode((line[2]))
#             example.append(InputExample(guid=guid, text=text, label=label, pos=pos))
#         return example
#
#     def get_train_examples(self, data_dir):
#         return self._create_example(
#             self._read_data('train'), 'train'
#         )
#
#     def get_dev_examples(self, data_dir):
#         return self._create_example(
#             self._read_data('dev'), 'dev'
#         )
#
#     def get_test_examples(self, data_dir):
#         return self._create_example(
#             self._read_data('test'), "test"
#         )
#
#     def get_labels(self):
#         """
#         based on ChinaDaily corpus
#         'X' is used to represent "##eer","##soo" and char not in vocab!
#         """
#         return ["X", "B-剧种", "I-剧种", "B-人名", "I-人名", "B-剧目", "I-剧目", "B-乐器", "I-乐器", "B-地点", "I-地点", "B-唱腔曲牌", "I-唱腔曲牌", "B-脚色行当", "I-脚色行当", "O", "[CLS]", "[SEP]"]
#
#
# def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, forward, backward, mode):
#     """
#     :param ex_index: example num
#     :param example:
#     :param label_list: all labels
#     :param max_seq_length:
#     :param tokenizer: WordPiece tokenization
#     :param mode:
#     :return: feature
#     IN this part we should rebuild input sentences to the following format.
#     example:[Jim,Hen,##son,was,a,puppet,##eer]
#     labels: [B,I,X,O,O,O,X]
#     """
#     label_map = dict()
#     # here start with zero which means that "[PAD]" is zero
#     # start with 1, 0 for paddding
#     for i, label in enumerate(label_list, 1):
#         label_map[label] = i
#
#     with open('./data/{}/label2id.pkl'.format(FLAGS.dataset), 'wb+') as wf:
#         pickle.dump(label_map, wf)
#
#     pos2id = dict()
#     with open('./data/{}/pos2id.pkl'.format(FLAGS.dataset), 'rb') as rf:
#         pos2id = pickle.load(rf)
#
#     textlist = example.text.split(' ')
#     labellist = example.label.split(' ')
#     # featurelist = [list(map(int, features.split('-'))) for features in example.feature.split(' ')]
#     poslist = example.pos.split(' ')
#     tokens = []
#     labels = []
#     poses = []
#     for i, (word, label, pos) in enumerate(zip(textlist, labellist, poslist)):
#         token = tokenizer.tokenize(word)
#         tokens.extend(token)
#         for m in range(len(token)):
#             if m == 0:
#                 labels.append(label)
#                 poses.append(pos)
#             else:
#                 labels.append('X')
#                 poses.append(0)
#     # only Account for [CLS] with "- 1".
#     # account for ending signal [SEP], with "- 2"
#     if len(tokens) >= max_seq_length - 1:
#         tokens = tokens[0: max_seq_length - 2]
#         labels = labels[0: max_seq_length - 2]
#         poses = poses[0: max_seq_length - 2]
#     ntokens = []
#     segment_ids = []
#     label_ids = []
#     pos_ids = []
#     # begin signal [CLS]
#     ntokens.append("[CLS]")
#     segment_ids.append(0)
#     label_ids.append(label_map["[CLS]"])
#     pos_ids.append(pos2id["[CLS]"])
#     for i, token in enumerate(tokens):
#         ntokens.append(token)
#         segment_ids.append(0)
#         label_ids.append(label_map[labels[i]])
#         pos_ids.append(int(poses[i]))
#     # ending signal [SEP]
#     ntokens.append("[SEP]")
#     segment_ids.append(0)
#     label_ids.append(label_map["[SEP]"])
#     pos_ids.append(pos2id["[SEP]"])
#
#     input_ids = tokenizer.convert_tokens_to_ids(ntokens)
#     input_mask = [1] * len(input_ids)
#     # use zero to padding sequence
#     while len(input_ids) < max_seq_length:
#         input_ids.append(0)
#         input_mask.append(0)
#         segment_ids.append(0)
#         label_ids.append(0)
#         pos_ids.append(0)
#         ntokens.append("**NULL**")
#     assert len(input_ids) == max_seq_length
#     assert len(input_mask) == max_seq_length
#     assert len(segment_ids) == max_seq_length
#     assert len(label_ids) == max_seq_length
#     assert len(pos_ids) == max_seq_length
#
#     if ex_index < 5:
#         logging.info("*** Example ***")
#         logging.info("guid: %s" % example.guid)
#         logging.info("tokens: %s" % " ".join(
#             [tokenization.printable_text(x) for x in tokens]))
#         logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#         logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#         logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#         logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
#         logging.info("pos_ids: %s" % " ".join([str(x) for x in pos_ids]))
#         logging.info("ntokens: %s" % " ".join([str(x) for x in ntokens]))
#
#     feature = InputFeatures(
#         input_ids=input_ids,
#         input_mask=input_mask,
#         segment_ids=segment_ids,
#         label_ids=label_ids,
#         pos_ids=pos_ids,
#         forward=forward,
#         backward=backward,
#     )
#     # we need ntokens because if we do predict it can help us return to original token.
#     return feature, ntokens, label_ids
#
#
# def file_based_convert_examples_to_features(
#         examples, label_list, max_seq_length, tokenizer, output_file, forward, backward, mode=None):
#     writer = tf.python_io.TFRecordWriter(path=output_file)
#     # writer = tf.io.TFRecordWriter(path=output_file)
#     batch_tokens = []
#     batch_labels = []
#     for ex_index, (example, a_fw, a_bw) in enumerate(zip(examples, forward, backward)):
#         if ex_index % 5000 == 0:
#             logging.info('Writing example {} of {}'.format(ex_index, len(examples)))
#         feature, ntokens, label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,
#                                                              a_fw, a_bw, mode)
#         batch_tokens.extend(ntokens)
#         batch_labels.extend(label_ids)
#
#         def create_int_feature(values):
#             f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
#             return f
#
#         def create_float_feature(values):
#             f = tf.train.Feature(float_list=tf.train.FloatList(value=values.flatten()))
#             return f
#
#         features = collections.OrderedDict()
#
#         features["input_ids"] = create_int_feature(feature.input_ids)
#         features["input_mask"] = create_int_feature(feature.input_mask)
#         features["segment_ids"] = create_int_feature(feature.segment_ids)
#         features["label_ids"] = create_int_feature(feature.label_ids)
#         features["pos_ids"] = create_int_feature(feature.pos_ids)
#         features["forward"] = create_float_feature(feature.forward)
#         features["backward"] = create_float_feature(feature.backward)
#         tf_example = tf.train.Example(features=tf.train.Features(feature=features))
#         writer.write(tf_example.SerializeToString())
#     writer.close()
#     return batch_tokens, batch_labels
#
#
# def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
#     name_to_features = {
#         "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
#         "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
#         "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
#         "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
#         "pos_ids": tf.FixedLenFeature([seq_length], tf.int64),
#         "forward": tf.VarLenFeature(tf.float32),
#         "backward": tf.VarLenFeature(tf.float32),
#     }
#
#     def _deocde_record(record, name_to_features):
#         example = tf.parse_single_example(record, name_to_features)
#         for name in list(example.keys()):
#             t = example[name]
#             if t.dtype == tf.int64:
#                 t = tf.to_int32(t)
#             if t.dtype == tf.float32:
#                 t = tf.sparse_tensor_to_dense(t)
#                 t = tf.reshape(t, [seq_length, seq_length])
#                 # print(t)
#                 # sess = tf.Session()
#                 # print(sess.run(t))
#             example[name] = t
#         return example
#
#     def input_fn(params):
#         batch_size = params["batch_size"]
#         d = tf.data.TFRecordDataset(input_file)
#         if is_training:
#             d = d.repeat()
#             d = d.shuffle(buffer_size=100)
#         d = d.apply(tf.contrib.data.map_and_batch(
#             lambda record: _deocde_record(record, name_to_features),
#             batch_size=batch_size,
#             drop_remainder=drop_remainder
#         ))
#         return d
#
#     return input_fn
#
#
# def hidden2tag(hiddenlayer, numclass):
#     # tf.keras.layers.Dense 封装了output = activation(tf.matmul(input, kernel) + bias)
#     # 相当于全连接层的线性变换
#     linear = tf.keras.layers.Dense(numclass, activation=None)
#     return linear(hiddenlayer)
#
#
# def softmax_layer(logits, labels, num_labels, mask):
#     logits = tf.reshape(logits, [-1, num_labels])
#     labels = tf.reshape(labels, [-1])
#     mask = tf.cast(mask, dtype=tf.float32)
#     one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
#     loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
#     loss *= tf.reshape(mask, [-1])
#     loss = tf.reduce_sum(loss)
#     total_size = tf.reduce_sum(mask)
#     total_size += 1e-12  # to avoid division by 0 for all-0 weights
#     loss /= total_size
#     # predict not mask we could filtered it in the prediction part.
#     probabilities = tf.math.softmax(logits, axis=-1)
#     predict = tf.math.argmax(probabilities, axis=-1)
#     return loss, predict
#
#
# def create_model(bert_config, is_training, input_ids, input_mask,
#                  segment_ids, label_ids, pos_ids, num_labels, forward, backward, use_one_hot_embeddings):
#     """
#     :param bert_config: bert 配置
#     :param is_training:
#     :param input_ids: 数据的idx 表示
#     :param input_mask:
#     :param segment_ids:
#     :param label_ids: 标签的idx 表示
#     :param pos_ids: 额外特征的idx表示 [batch_size, seq_length]
#     :param num_labels: 类别数量
#     :param use_one_hot_embeddings:
#     :return:
#     """
#     # 使用数据加载BertModel,获取对应的字embedding
#     model = modeling.BertModel(
#         config=bert_config,
#         is_training=is_training,
#         input_ids=input_ids,
#         input_mask=input_mask,
#         token_type_ids=segment_ids,
#         use_one_hot_embeddings=use_one_hot_embeddings
#     )
#     # [batch_size, seq_length, embedding_size]
#     # use model.get_sequence_output() to get token-level output
#     output_embeddings = model.get_sequence_output()
#
#     embeddings = output_embeddings
#     print('embedding shape',embeddings.shape)
#
#     if FLAGS.use_pos:
#         pos2id = {}
#         with open('./data/{}/pos2id.pkl'.format(FLAGS.dataset), 'rb') as rf:
#             pos2id = pickle.load(rf)
#         num_pos = len(pos2id)
#         pos_embeddings = tf.one_hot(pos_ids, num_pos, axis=-1)  # 维度 pos_ids*num_pos
#         embeddings = tf.concat([embeddings, pos_embeddings], -1)  # 加长embedding向量
#         print(embeddings.shape)
#
#     if FLAGS.bilstm or FLAGS.crf:
#         '''
#         used = tf.sign(tf.abs(input_ids))
#         lengths = tf.reduce_sum(used, reduction_indices=1)
#         '''
#         # [batch_size] 大小的向量，包含了当前batch中的序列长度
#         lengths = tf.reduce_sum(input_mask, axis=1)
#         max_seq_length = embeddings.shape[1].value
#
#         if FLAGS.gcn == 0:
#             bilstm_crf = BiLSTM_GCN_CRF3(embedded_chars=embeddings, lstm_size=FLAGS.lstm_size, cell_type=FLAGS.cell,
#                                 num_layers=FLAGS.num_layers,
#                                 dropout_rate=FLAGS.dropout_rate, num_labels=num_labels,
#                                 max_seq_length=max_seq_length,
#                                 labels=label_ids,
#                                 lengths=lengths, is_training=is_training, bilstm=FLAGS.bilstm, crf=FLAGS.crf, forward=forward, backward=backward)
#             loss, predict = bilstm_crf.add_bilstm_crf_layer()
#         elif FLAGS.gcn == 1:
#             bilstm_crf = BiLSTM_GCN_CRF(embedded_chars=embeddings, lstm_size=FLAGS.lstm_size, cell_type=FLAGS.cell,
#                                 num_layers=FLAGS.num_layers,
#                                 dropout_rate=FLAGS.dropout_rate, num_labels=num_labels,
#                                 max_seq_length=max_seq_length,
#                                 labels=label_ids,
#                                 lengths=lengths, is_training=is_training, bilstm=FLAGS.bilstm, crf=FLAGS.crf, forward=forward, backward=backward)
#             loss, predict = bilstm_crf.add_bilstm_crf_layer()
#         elif FLAGS.gcn == 2:
#             bilstm_crf = BiLSTM_GCN_CRF2(embedded_chars=embeddings, lstm_size=FLAGS.lstm_size, cell_type=FLAGS.cell,
#                                 num_layers=FLAGS.num_layers,
#                                 dropout_rate=FLAGS.dropout_rate, num_labels=num_labels,
#                                 max_seq_length=max_seq_length,
#                                 labels=label_ids,
#                                 lengths=lengths, is_training=is_training, bilstm=FLAGS.bilstm, crf=FLAGS.crf, forward=forward, backward=backward)
#             loss, predict = bilstm_crf.add_bilstm_crf_layer()
#     else:
#         if is_training:
#             embeddings = tf.nn.dropout(embeddings, keep_prob=0.9)
#         logits = hidden2tag(embeddings, num_labels)
#         logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
#         loss, predict = softmax_layer(logits, label_ids, num_labels, input_mask)
#
#     return (loss, predict)
#
#
# def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
#                      num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
#     def model_fn(features, labels, mode, params):
#         logging.info("*** Features ***")
#         for name in sorted(features.keys()):
#             logging.info("  name = {}, shape = {}".format(name, features[name].shape))
#         input_ids = features["input_ids"]
#         input_mask = features["input_mask"]
#         segment_ids = features["segment_ids"]
#         label_ids = features["label_ids"]
#         pos_ids = features["pos_ids"]
#         forward = features["forward"]
#         backward = features["backward"]
#         is_training = (mode == tf.estimator.ModeKeys.TRAIN)
#
#         (total_loss, pred_ids) = create_model(
#             bert_config, is_training, input_ids, input_mask, segment_ids,
#             label_ids, pos_ids, num_labels, forward, backward, use_one_hot_embeddings)
#
#         vars = tf.trainable_variables()
#         scaffold_fn = None
#         if init_checkpoint:
#             (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
#                 vars, init_checkpoint)
#             tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
#             if use_tpu:
#                 def tpu_scaffold():
#                     tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
#                     return tf.train.Scaffold()
#
#                 scaffold_fn = tpu_scaffold
#             else:
#                 tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
#         logging.info("**** Trainable Variables ****")
#
#         for var in vars:
#             init_string = ""
#             if var.name in initialized_variable_names:
#                 init_string = ", *INIT_FROM_CKPT*"
#             logging.info("  name = {}, shape = {} {}".format(var.name, var.shape, init_string))
#
#         output_spec = None
#         if mode == tf.estimator.ModeKeys.TRAIN:
#             train_op = optimization.create_optimizer(
#                 total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
#             output_spec = tf.contrib.tpu.TPUEstimatorSpec(
#                 mode=mode,
#                 loss=total_loss,
#                 train_op=train_op,
#                 scaffold_fn=scaffold_fn)
#         elif mode == tf.estimator.ModeKeys.EVAL:
#             # 针对NER ,进行了修改
#             def metric_fn(label_ids, pred_ids):
#                 try:
#                     # confusion matrix
#                     cm = tf_metrics.streaming_confusion_matrix(label_ids, pred_ids, num_labels, weights=input_mask)
#                     return {
#                         "confusion_matrix": cm
#                     }
#                 except Exception as e:
#                     logging.error(str(e))
#
#             eval_metrics = (metric_fn, [label_ids, pred_ids])
#
#             output_spec = tf.contrib.tpu.TPUEstimatorSpec(
#                 mode=mode, loss=total_loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn
#             )
#         else:
#             output_spec = tf.contrib.tpu.TPUEstimatorSpec(
#                 mode=mode, predictions=pred_ids, scaffold_fn=scaffold_fn)
#         return output_spec
#
#     return model_fn
#
#
# def main(_):
#     # tf.logging.set_verbosity(tf.logging.INFO)
#
#     for name, value in FLAGS.__flags.items():
#         logging.info('{} = {}'.format(name, value.value))
#
#     processors = {
#         "ner": NerProcessor
#     }
#
#     data_dir = './data/{}/'.format(FLAGS.dataset)
#
#     bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
#
#     if FLAGS.max_seq_length > bert_config.max_position_embeddings:
#         raise ValueError(
#             "Cannot use sequence length {} because the BERT model, was only trained up to sequence length {}".format(
#                 FLAGS.max_seq_length, bert_config.max_position_embeddings))
#
#     # tf.gfile.MakeDirs(FLAGS.output_dir)
#
#     task_name = FLAGS.task_name.lower()
#     if task_name not in processors:
#         raise ValueError("Task not found: {}".format(task_name))
#
#     processor = processors[task_name]()
#
#     label_list = processor.get_labels()
#
#     tokenizer = tokenization.FullTokenizer(
#         vocab_file=FLAGS.vocab_file,
#         do_lower_case=FLAGS.do_lower_case
#     )
#     tpu_cluster_resolver = None
#     if FLAGS.use_tpu and FLAGS.tpu_name:
#         tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
#             FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
#
#     is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
#
#     run_config = tf.contrib.tpu.RunConfig(
#         cluster=tpu_cluster_resolver,
#         master=FLAGS.master,
#         model_dir=FLAGS.output_dir,
#         save_checkpoints_steps=FLAGS.save_checkpoints_steps,
#         tpu_config=tf.contrib.tpu.TPUConfig(
#             iterations_per_loop=FLAGS.iterations_per_loop,
#             num_shards=FLAGS.num_tpu_cores,
#             per_host_input_for_training=is_per_host
#         )
#     )
#
#     train_examples = None
#     num_train_steps = None
#     num_warmup_steps = None
#
#     if FLAGS.do_train:
#         train_examples = processor.get_train_examples(data_dir)
#         num_train_steps = int(
#             len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
#         num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
#
#     # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法,
#     # 并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程;
#     # tf 新的架构方法，通过定义model_fn 函数，定义模型,
#     # 然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。
#
#     model_fn = model_fn_builder(
#         bert_config=bert_config,
#         num_labels=len(label_list) + 1,
#         init_checkpoint=FLAGS.init_checkpoint,
#         learning_rate=FLAGS.learning_rate,
#         num_train_steps=num_train_steps,
#         num_warmup_steps=num_warmup_steps,
#         use_tpu=FLAGS.use_tpu,
#         use_one_hot_embeddings=FLAGS.use_tpu)
#
#     estimator = tf.contrib.tpu.TPUEstimator(
#         use_tpu=FLAGS.use_tpu,
#         model_fn=model_fn,
#         config=run_config,
#         train_batch_size=FLAGS.train_batch_size,
#         eval_batch_size=FLAGS.eval_batch_size,
#         predict_batch_size=FLAGS.predict_batch_size)
#
#
#     if FLAGS.do_train:
#         # 图模型
#         a = data_dir+'train.txt'
#         sentences = aux1.get_all_sentences(a)
#         data = aux1.get_data_from_sentences(sentences)
#
#         # forward = []
#         # backward = []
#         # for item in data:
#         #     words = item[0]
#         #     sentence = aux1.create_full_sentence(words)
#         #     A_fw, A_bw = aux1.create_graph_from_sentence_and_word_vectors(sentence, FLAGS.max_seq_length)
#         #     forward.append(A_fw)
#         #     backward.append(A_bw)
#
#         # 1. 将数据转化为tf_record 数据
#         train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
#         # print('***train_file***', train_file)
#         # print('-----------', len(forward))
#         # _, _ = file_based_convert_examples_to_features(
#         #     train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, forward, backward)
#         logging.info("***** Running training *****")
#         logging.info("  Num examples = %d", len(train_examples))
#         logging.info("  Batch size = %d", FLAGS.train_batch_size)
#         logging.info("  Num steps = %d", num_train_steps)
#
#         # 2.读取record 数据，组成batch
#         train_input_fn = file_based_input_fn_builder(
#             input_file=train_file,
#             seq_length=FLAGS.max_seq_length,
#             is_training=True,
#             drop_remainder=True)
#
#
#         estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
#
#     if FLAGS.do_eval:
#         # 图模型
#         a = data_dir+'dev.txt'
#         sentences = aux1.get_all_sentences(a)
#         data = aux1.get_data_from_sentences(sentences)
#         # buckets = aux1.bin_data_into_buckets(data, FLAGS.train_batch_size)
#         # random_buckets = sorted(buckets, key=lambda x: random.random())
#         forward = []
#         backward = []
#         for item in data:
#             words = item[0]
#             sentence = aux1.create_full_sentence(words)
#             A_fw, A_bw = aux1.create_graph_from_sentence_and_word_vectors(sentence, FLAGS.max_seq_length)
#             forward.append(A_fw)
#             backward.append(A_bw)
#         eval_examples = processor.get_dev_examples(data_dir)
#         eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
#         _, _ = file_based_convert_examples_to_features(
#             eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, forward, backward)
#
#         logging.info("***** Running evaluation *****")
#         logging.info("  Num examples = %d", len(eval_examples))
#         logging.info("  Batch size = %d", FLAGS.eval_batch_size)
#
#         eval_steps = None
#         eval_drop_remainder = True if FLAGS.use_tpu else False
#         if FLAGS.use_tpu:
#             eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
#
#         eval_input_fn = file_based_input_fn_builder(
#             input_file=eval_file,
#             seq_length=FLAGS.max_seq_length,
#             is_training=False,
#             drop_remainder=eval_drop_remainder)
#
#         result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
#         output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
#         output_eval_cm = os.path.join(FLAGS.output_dir, "eval_results_cm.txt")
#
#         with open(output_eval_file, "a+", encoding='utf-8') as writer:
#             logging.info("***** Eval results *****")
#             writer.write('data:\t{}\n'.format(FLAGS.dataset))
#             writer.write('model:\t{}\n'.format(FLAGS.init_checkpoint))
#             writer.write('loss:\t{}\n'.format(result['loss']))
#             writer.write('global_step:\t{}\n'.format(result['global_step']))
#             confusion_matrix = result.get("confusion_matrix", None)
#             with open(output_eval_cm, "a+", encoding='utf-8') as fw_cm:
#                 fw_cm.write('data:\t{}\n'.format(FLAGS.dataset))
#                 fw_cm.write('model:\t{}\n'.format(FLAGS.init_checkpoint))
#                 for row in confusion_matrix:
#                     for col in row:
#                         fw_cm.write('{:1f}\t'.format(col))
#                     fw_cm.write('\n')
#             try:
#                 precisions, recalls, fs, acc, kappa = tf_metrics.calculate(confusion_matrix)
#                 writer.write('Precision: {}\n'.format('\t'.join([str(p) for p in precisions])))
#                 writer.write('Recall: {}\n'.format('\t'.join([str(r) for r in recalls])))
#                 writer.write('F1: {}\n'.format('\t'.join([str(f) for f in fs])))
#                 writer.write('Acc: {}\n'.format(acc))
#                 writer.write('Kappa: {}\n'.format(kappa))
#             except Exception as e:
#                 logging.error(str(e))
#
#     if FLAGS.do_test:
#         with open('./data/{}/label2id.pkl'.format(FLAGS.dataset), 'rb') as rf:
#             label2id = pickle.load(rf)
#             id2label = {value: key for key, value in label2id.items()}
#         predict_examples = processor.get_test_examples(data_dir)
#         # 图模型
#         a = data_dir+'test.txt'
#         sentences = aux1.get_all_sentences(a)
#         data = aux1.get_data_from_sentences(sentences)
#         # buckets = aux1.bin_data_into_buckets(data, FLAGS.train_batch_size)
#         # random_buckets = sorted(buckets, key=lambda x: random.random())
#         forward = []
#         backward = []
#         # for bucket in buckets:
#         for item in data:
#             words = item[0]
#             sentence = aux1.create_full_sentence(words)
#             A_fw, A_bw = aux1.create_graph_from_sentence_and_word_vectors(sentence, FLAGS.max_seq_length)
#             forward.append(A_fw)
#             backward.append(A_bw)
#
#         predict_file = os.path.join(FLAGS.output_dir, "test.tf_record")
#         batch_tokens, batch_labels = file_based_convert_examples_to_features(
#             predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file, forward, backward, mode="test")
#
#         logging.info("***** Running Test*****")
#         logging.info("  Num examples = %d", len(predict_examples))
#         logging.info("  Batch size = %d", FLAGS.predict_batch_size)
#
#         predict_drop_remainder = True if FLAGS.use_tpu else False
#         if FLAGS.use_tpu:
#             # Warning: According to tpu_estimator.py Prediction on TPU is an experimental feature and hence not supported here
#             raise ValueError("Prediction in TPU not supported")
#
#         predict_input_fn = file_based_input_fn_builder(
#             input_file=predict_file,
#             seq_length=FLAGS.max_seq_length,
#             is_training=False,
#             drop_remainder=predict_drop_remainder)
#
#         result = estimator.predict(input_fn=predict_input_fn)
#         output_test_file = os.path.join(FLAGS.output_dir, "test_label.txt")
#         result_test_file = os.path.join(FLAGS.output_dir, "test_result.txt")
#
#         # here if the tag is "X" means it belong to its before token for convenient evaluate use
#         def Writer(output_test_file, result, batch_tokens, batch_labels, id2label):
#             with open(output_test_file, 'w+', encoding='UTF-8') as wf:
#
#                 if FLAGS.bilstm:
#                     predictions = []
#                     for _, pred in enumerate(result):
#                         predictions.extend(pred)
#                 else:
#                     predictions = result
#
#                 for i, prediction in enumerate(predictions):
#                     token = batch_tokens[i]
#                     if prediction == 0:
#                         continue
#                     predict = id2label[prediction]
#                     true_label = id2label[batch_labels[i]]
#                     if token in ['[CLS]', '[SEP]']:
#                         continue
#                     line = "{}\t{}\t{}\n".format(token, true_label, predict)
#                     wf.write(line)
#
#         Writer(output_test_file, result, batch_tokens, batch_labels, id2label)
#
#         test_report = conlleval.return_report(output_test_file)
#         logging.info(''.join(test_report))
#         with open(result_test_file, 'a+', encoding='UTF-8') as wf:
#             wf.write('data:\t{}\n'.format(FLAGS.dataset))
#             wf.write('model:\t{}\n'.format(FLAGS.init_checkpoint))
#             wf.write(''.join(test_report))
#
#
# if __name__ == '__main__':
#     flags.mark_flag_as_required("dataset")
#     flags.mark_flag_as_required("vocab_file")
#     flags.mark_flag_as_required("bert_config_file")
#     flags.mark_flag_as_required("output_dir")
#     if not os.path.exists(FLAGS.output_dir):
#         tf.gfile.MakeDirs(FLAGS.output_dir)
#     LOG_SETTINGS = {
#         'format': '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#         'datefmt': '%Y-%m-%d %H:%M:%S',
#     }
#     FILE_NAME = os.path.join(FLAGS.output_dir, '{}.log'.format('train' if FLAGS.do_train else 'test'))
#     logging.basicConfig(
#         handlers=[logging.FileHandler(FILE_NAME, encoding="utf-8", mode='a')],
#         level=logging.INFO,
#         **LOG_SETTINGS
#     )
#     tf.app.run()
#     # tf.compat.v1.app.run()
















