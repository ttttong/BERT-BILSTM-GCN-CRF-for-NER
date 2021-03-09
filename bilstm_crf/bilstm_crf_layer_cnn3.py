import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np

import pickle

flags = tf.flags
FLAGS = flags.FLAGS


class BiLSTM_CRF_CNN:
    def __init__(self, embedded_chars, lstm_size, cell_type, num_layers, dropout_rate,
                 num_labels, max_seq_length, labels, lengths, is_training, bilstm=True, crf=True, cnn=True):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param lstm_size: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param dropout_rate: droupout rate
        :param num_labels: 标签数量
        :param max_seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.lstm_size = lstm_size
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.max_seq_length = max_seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.embedding_size = embedded_chars.shape[-1].value
        self.kernel_size = 50
        self.is_training = is_training
        self.crf = crf
        self.cnn = cnn
        self.bilstm = bilstm
        self.initializer = initializers.xavier_initializer()

    def _construct_cell(self):
        cell = None
        if self.cell_type == 'lstm':
            cell = rnn.LSTMCell(self.lstm_size)
        elif self.cell_type == 'gru':
            cell = rnn.GRUCell(self.lstm_size)

        if self.dropout_rate is not None:
            cell = rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_rate)
        return cell

    def bilstm_layer(self, pos_ids):
        with tf.variable_scope('bilstm_layer'):
            cell_fw = self._construct_cell()
            cell_bw = self._construct_cell()

            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)

            output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                        self.embedded_chars, dtype=tf.float32)
            output = tf.concat(output, axis=2)
            '''
                 bilstm输出与pos拼接
            '''
            # # 把特征加在lstm第一层
            # if FLAGS.use_pos:
            #     pos2id = {}
            #     with open('./data/{}/pos2id.pkl'.format(FLAGS.dataset), 'rb') as rf:
            #         pos2id = pickle.load(rf)
            #     num_pos = len(pos2id)
            #     pos_embeddings = tf.one_hot(pos_ids, num_pos, axis=-1)  # 维度 pos_ids*num_pos
            #     output = tf.concat([output,pos_embeddings], axis=2)
        return output


    def cnn_layer(self, lstm_output):
        with tf.variable_scope('cnn_layer'):
            # filter格式[filter_width, embedding, kernel]
            filter = tf.get_variable(
                "filter",
                shape=[5, self.embedding_size, self.kernel_size],
                dtype=np.float32,
                initializer=self.initializer)
            output2 = tf.nn.conv1d(self.embedded_chars, filters=filter, padding="SAME", stride=1)
            print('cnn_output', output2.shape)
            output = tf.concat([lstm_output,output2], axis=2)
            print('lstm+crf', output.shape)
        return output


    ## 矩阵变化
    def project_bilstm_layer(self, cnn_lstm_output, pos_ids):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        # tf.contrib.layers.xavier_initializer
        # 初始化权重矩阵
        with tf.variable_scope('project_bilstm_layer'):
            # 把特征加在lstm层
            # if FLAGS.use_pos:
            #     pos2id = {}
            #     with open('./data/{}/pos2id.pkl'.format(FLAGS.dataset), 'rb') as rf:
            #         pos2id = pickle.load(rf)
            #     num_pos = len(pos2id)
            #     W = tf.get_variable('W', shape=[self.lstm_size * 2+self.kernel_size+num_pos, self.lstm_size],
            #                         dtype=tf.float32, initializer=self.initializer)
            #     b = tf.get_variable("b", shape=[self.lstm_size], dtype=tf.float32,
            #                             initializer=tf.zeros_initializer())
            #     output = tf.reshape(cnn_lstm_output, shape=[-1, self.lstm_size * 2+self.kernel_size+num_pos])
            #     print('--------output:',output.shape)
            #     hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            #     print('--------hidden:',output.shape)

            # 把特征加在softmax层
            W = tf.get_variable('W', shape=[self.lstm_size * 2+self.kernel_size, self.lstm_size],
                                dtype=tf.float32, initializer=self.initializer)
            b = tf.get_variable("b", shape=[self.lstm_size], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            output = tf.reshape(cnn_lstm_output, shape=[-1, self.lstm_size * 2+self.kernel_size])
            hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            if FLAGS.use_pos:
                pos2id = {}
                with open('./data/{}/pos2id.pkl'.format(FLAGS.dataset), 'rb') as rf:
                    pos2id = pickle.load(rf)
                num_pos = len(pos2id)
                pos_embeddings = tf.one_hot(pos_ids, num_pos, axis=-1)  # 维度 pos_ids*num_pos
                pos_embeddings = tf.reshape(pos_embeddings, shape=[-1, num_pos])
                hidden = tf.concat([hidden, pos_embeddings], axis=1)

        # project to score of tags
        with tf.variable_scope("logits"):
            # 如果加在第一层 w为self.lstm_size/加在softmax层 w为self.lstm_size + num_pos
            W = tf.get_variable("W", shape=[self.lstm_size + num_pos, self.num_labels],
                                dtype=tf.float32, initializer=self.initializer)

            b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            # tf.nn.xw_plus_b(x, weights, biases.) 相当于 matmul(x, weights) + biases.
            pred = tf.nn.xw_plus_b(hidden, W, b)
            print('--------pred', pred.shape)
            a = tf.reshape(pred, [-1, self.max_seq_length, self.num_labels])
            print('---------pred.reshape', a)
        return tf.reshape(pred, [-1, self.max_seq_length, self.num_labels])

    def project_crf_layer(self, embedding_chars):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project_crf_layer"):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_size, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars,
                                    shape=[-1, self.embedding_size])  # [batch_size, embedding_size]
                pred = tf.nn.xw_plus_b(output, W, b)
            return tf.reshape(pred, [-1, self.max_seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializer)
            if self.labels is None:
                return None, trans
            else:
                log_likelihood, trans = crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.labels,
                    transition_params=trans,
                    sequence_lengths=self.lengths)
                return tf.reduce_mean(-log_likelihood), trans

    def softmax_layer(self, logits):
        pred_ids = tf.argmax(logits, axis=-1)
        pred_ids = tf.cast(pred_ids, tf.int32)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
        mask = tf.sequence_mask(self.lengths)
        losses = tf.boolean_mask(losses, mask)
        loss = tf.reduce_mean(losses)
        return pred_ids, loss

    def add_bilstm_crf_layer(self, pos_ids):
        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        if not self.bilstm:
            # project layer
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            # bilstm_layer
            lstm_output = self.bilstm_layer(pos_ids)
            # cnn_layer
            cnn_output = self.cnn_layer(lstm_output)
            #link layer

            # project layer
            logits = self.project_bilstm_layer(cnn_output, pos_ids)

        if not self.crf:
            # softmax layer
            pred_ids, loss = self.softmax_layer(logits)
        else:
            # crf_layer
            loss, trans = self.crf_layer(logits)
            # CRF decode, pred_ids 是一条最大概率的标注路径
            pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)

        return (loss, pred_ids)