from ltp import LTP
import tensorflow as tf
import numpy as np
ltp = LTP()

# 截断128维度句子
class Parser:

    def __init__(self, sentence, max_seq_length):
        self.sentence = sentence
        self.max_seq_length = max_seq_length

    def execute(self):
        b = []
        b.append(self.sentence)
        seg, hidden = ltp.seg(b)
        dep = ltp.dep(hidden)
        word_tree = dep[0]
        a = seg[0]
        word2char, j = {0: [0]}, 1
        for i, word in enumerate(a, start=1):
            word2char[i] = list(range(j, j + len(word)))
            j += len(word)
        # # convert tree
        # char_tree = []
        # for arc in word_tree:
        #     dep, head, pos = arc
        #     dep_char, head_char = word2char[dep], word2char[head]
        #     for d in dep_char:
        #         for h in head_char:
        #             char_tree.append((d, h))
        # bulid matrix
        import numpy as np
        from itertools import product
        sent_len = sum((len(word) for word in a)) + 1
        char_tree_matrix = np.zeros((self.max_seq_length, self.max_seq_length))
        for arc in word_tree:
            dep, head, pos = arc
            dep_char, head_char = word2char[dep], word2char[head]
            # ind1, ind2 = zip(*product(dep_char, head_char))
            for d in dep_char:
                for h in head_char:
                    if d <= self.max_seq_length and h <= self.max_seq_length:
                        if h != 0:
                            char_tree_matrix[d - 1, h - 1] = 1

        # # 新加的
        # # 1.加单位矩阵
        # num_nodes = 128
        # identity = tf.eye(num_nodes)
        # identity = tf.expand_dims(identity, axis=0)
        # char_tree_matrix = identity + char_tree_matrix

        return char_tree_matrix

    def execute_backward(self):
        b = []
        b.append(self.sentence)
        seg, hidden = ltp.seg(b)
        dep = ltp.dep(hidden)
        word_tree = dep[0]
        a = seg[0]
        word2char, j = {0: [0]}, 1
        for i, word in enumerate(a, start=1):
            word2char[i] = list(range(j, j + len(word)))
            j += len(word)
        # # convert tree
        # char_tree = []
        # for arc in word_tree:
        #     dep, head, pos = arc
        #     dep_char, head_char = word2char[dep], word2char[head]
        #     for d in dep_char:
        #         for h in head_char:
        #             char_tree.append((d, h))
        # bulid matrix
        import numpy as np
        from itertools import product
        sent_len = sum((len(word) for word in a)) + 1
        char_tree_matrix = np.zeros((self.max_seq_length, self.max_seq_length))
        for arc in word_tree:
            dep, head, pos = arc
            dep_char, head_char = word2char[dep], word2char[head]
            # ind1, ind2 = zip(*product(dep_char, head_char))
            for d in dep_char:
                for h in head_char:
                    if d <= self.max_seq_length and h <= self.max_seq_length:
                        if h != 0:
                            char_tree_matrix[h-1, d-1] = 1

        # # 新加的
        # # 1.加单位矩阵
        # num_nodes = 128
        # identity = tf.eye(num_nodes)
        # identity = tf.expand_dims(identity, axis=0)
        # char_tree_matrix = identity + char_tree_matrix

        return char_tree_matrix

a = '高腔是中国戏曲四大声腔之一。'
b = Parser(a,6)
print(b.execute())