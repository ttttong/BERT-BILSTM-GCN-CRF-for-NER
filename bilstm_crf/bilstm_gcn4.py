import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import aux1

import pickle
flags = tf.flags
FLAGS = flags.FLAGS
import numpy as np
import aux1

class BiLSTM_GCN_CRF2:
    def __init__(self, embedded_chars, lstm_size, cell_type, num_layers, dropout_rate,
                 num_labels, max_seq_length, labels, lengths, is_training, forward, backward, bilstm=True, crf=True):
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
        :param sentence: 图结构
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
        self.is_training = is_training
        self.crf = crf
        self.bilstm = bilstm
        self.initializer = initializers.xavier_initializer()
        self.forward = forward
        self.backward = backward

    def _construct_cell(self):
        cell = None
        if self.cell_type == 'lstm':
            cell = rnn.LSTMCell(self.lstm_size)
        elif self.cell_type == 'gru':
            cell = rnn.GRUCell(self.lstm_size)

        if self.dropout_rate is not None:
            cell = rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_rate)
        return cell

    def bilstm_layer(self):
        with tf.variable_scope('bilstm_layer'):
            cell_fw = self._construct_cell()
            cell_bw = self._construct_cell()

            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
            print('self.embedded_chars', self.embedded_chars.shape)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                        self.embedded_chars, dtype=tf.float32)
            output = tf.concat(output, axis=2)
        return output

    # gcn layer
    def _add_identity(self, A):
        num_nodes = 128
        identity = tf.eye(num_nodes)
        identity = tf.expand_dims(identity, axis=0)
        return identity + A

    # gcn layer
    def GCN_layer(self, hidden, A_fw, A_bw):

        # Atilde_fw = tf.placeholder(tf.float32, shape=(None, None, None), name="Atilde_fw")
        # Atilde_bw = tf.placeholder(tf.float32, shape=(None, None, None), name="Atilde_bw")
        #原来的
        Atilde_fw = self._add_identity(A_fw)
        Atilde_bw = self._add_identity(A_bw)
        # Atilde_fw = A_fw
        # Atilde_bw = A_bw


        W0_fw = tf.Variable(tf.random_uniform([self.embedding_size, self.lstm_size], 0, 0.1), name='W0_fw')
        b0_fw = tf.Variable(tf.random_uniform([self.lstm_size], -0.1, 0.1), name='b0_fw')
        left_X1_projection_fw = lambda x: tf.matmul(x, W0_fw) + b0_fw
        left_X1_fw = tf.map_fn(left_X1_projection_fw, self.embedded_chars)
        # left_X1_fw = tf.transpose(left_X1_fw, perm=[1, 0, 2], name='left_X1_fw')
        X1_fw = tf.nn.relu(tf.matmul(Atilde_fw, left_X1_fw))
        # X1_fw = tf.transpose(X1_fw, perm=[1, 0, 2])
        # bw
        W0_bw = tf.Variable(tf.random_uniform([self.embedding_size, self.lstm_size], 0, 0.1), name='W0_bw')
        b0_bw = tf.Variable(tf.random_uniform([self.lstm_size], -0.1, 0.1), name='b0_bw')
        left_X1_projection_bw = lambda x: tf.matmul(x, W0_bw) + b0_bw
        left_X1_bw = tf.map_fn(left_X1_projection_bw, self.embedded_chars)
        # left_X1_bw = tf.transpose(left_X1_bw, perm=[1, 0, 2], name='left_X1_bw')
        X1_bw = tf.nn.relu(tf.matmul(Atilde_bw, left_X1_bw))
        # X1_bw = tf.transpose(X1_bw, perm=[1, 0, 2])
        output = tf.concat(values=[X1_fw, X1_bw], axis=2)

        #第二层GCN
        W1_fw = tf.Variable(tf.random_uniform([self.lstm_size*2, self.lstm_size], 0, 0.1), name='W1_fw')
        b1_fw = tf.Variable(tf.random_uniform([self.lstm_size], -0.1, 0.1), name='b1_fw')
        left_X2_projection_fw = lambda x: tf.matmul(x, W1_fw) + b1_fw
        left_X2_fw = tf.map_fn(left_X2_projection_fw, output)
        # left_X1_fw = tf.transpose(left_X1_fw, perm=[1, 0, 2], name='left_X1_fw')
        X2_fw = tf.nn.relu(tf.matmul(Atilde_fw, left_X2_fw))
        # X1_fw = tf.transpose(X1_fw, perm=[1, 0, 2])
        # bw
        W1_bw = tf.Variable(tf.random_uniform([self.lstm_size*2, self.lstm_size], 0, 0.1), name='W1_bw')
        b1_bw = tf.Variable(tf.random_uniform([self.lstm_size], -0.1, 0.1), name='b1_bw')
        left_X2_projection_bw = lambda x: tf.matmul(x, W1_bw) + b1_bw
        left_X2_bw = tf.map_fn(left_X2_projection_bw, output)
        # left_X1_bw = tf.transpose(left_X1_bw, perm=[1, 0, 2], name='left_X1_bw')
        X2_bw = tf.nn.relu(tf.matmul(Atilde_bw, left_X2_bw))
        # X1_bw = tf.transpose(X1_bw, perm=[1, 0, 2])
        output = tf.concat(values=[X2_fw, X2_bw], axis=2)


        output = tf.concat(values=[output, hidden], axis=2)
        return output

    def project_bilstm_layer(self, gcn_output, pos_ids):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        # tf.contrib.layers.xavier_initializer
        # 初始化权重矩阵
        with tf.variable_scope('project_bilstm_layer'):
            W = tf.get_variable('W', shape=[self.lstm_size * 4, self.lstm_size],
                                dtype=tf.float32, initializer=self.initializer)
            b = tf.get_variable("b", shape=[self.lstm_size], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            output = tf.reshape(gcn_output, shape=[-1, self.lstm_size * 4])
            hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))


        # project to score of tags
        with tf.variable_scope("logits"):
            W = tf.get_variable("W", shape=[self.lstm_size+num_pos, self.num_labels],
                                dtype=tf.float32, initializer=self.initializer)

            b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            # tf.nn.xw_plus_b(x, weights, biases.) 相当于 matmul(x, weights) + biases.
            pred = tf.nn.xw_plus_b(hidden, W, b)
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
            lstm_output = self.bilstm_layer()
            # gcn_layer
            gcn_output = self.GCN_layer(lstm_output, A_fw=self.forward, A_bw=self.backward)
            # project layer
            logits = self.project_bilstm_layer(gcn_output, pos_ids)

        if not self.crf:
            # softmax layer
            pred_ids, loss = self.softmax_layer(logits)
        else:
            # crf_layer
            loss, trans = self.crf_layer(logits)
            # CRF decode, pred_ids 是一条最大概率的标注路径
            pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)

        return (loss, pred_ids)
