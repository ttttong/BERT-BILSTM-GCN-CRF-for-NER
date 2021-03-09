"""Created by PeterLee, on Dec. 17."""
import pickle, random
import numpy as np
import logging


def build_embedding_source(source_path, vocab_path, embedding_path):
    n_char, n_dim = 0, 0
    char2id = {}
    with open(source_path, encoding='utf-8') as fr:
        first_line = fr.readline()
        n_char = int(first_line.strip().split()[0])
        n_dim = int(first_line.strip().split()[1])
        logging.info('n_char: {}, n_dim: {}'.format(n_char, n_dim))
        char2id['<UNK>'] = 0
        embeddings = np.float32(np.random.uniform(-0.25, 0.25, (1, n_dim)))
        new_line = fr.readline()
        while(new_line):
            elements = new_line.strip().split()
            char = elements[0]
            embedding = np.array(
                [float(x) for x in elements[1:]]
            ).astype(np.float32)
            char2id[char] = len(char2id) + 1
            embeddings = np.concatenate((embeddings, np.reshape(embedding, (1, n_dim))))
            new_line = fr.readline()
        logging.info('shape of embeddings: {}'.format(embeddings.shape))
        logging.info('size of vocabulary: {}'.format(len(char2id)))
    with open(embedding_path, 'w+') as fw:
        np.savetxt(fw, embeddings, delimiter=' ', newline='\n')
    with open(vocab_path, 'wb+') as fw:
        pickle.dump(char2id, fw)


def read_corpus(corpus_path):
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    count = 0
    for line in lines:
        if line != '\n':
            # print(line)
            # count = count+1
            # print(count)
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    return data


def vocab_build(vocab_path, corpus_path, min_count):
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0
    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        # # Inspecting the str whether combine by number
        # if word.isdigit():
        #     word = '<NUM>'
        # # Judging the english
        # elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
        #     word = '<ENG>'
        if word not in word2id:
            # Chinese
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    return word2id


def random_embedding(vocab, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    # padding
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        # The length of seq
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    if shuffle:
        # Random data
        random.shuffle(data)
    seqs, labels = [], []

    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []
        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        # Return an iteration
        yield seqs, labels
