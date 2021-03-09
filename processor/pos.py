import os
import pickle
import pynlpir
from processor.data import read_corpus


# file_name = 'train'
# file_input = '../data/MSRA/{}.txt'.format(file_name)
# file_output = '../data/MSRA-pos/{}.txt'.format(file_name)
# file_pos_pkl = '../data/MSRA-pos/pos2id.pkl'
# file_pos_txt = '../data/MSRA-pos/pos2id.txt'

file_name = 'test'
file_input = '../data/MSRA/{}.txt'.format(file_name)
file_output = '../data/MSRA-pos/{}.txt'.format(file_name)
file_pos_pkl = '../data/opera2-pos/pos2id.pkl'
file_pos_txt = '../data/opera2-pos/pos2id.txt'


reserve_pos_list = ['noun', 'verb', 'adjective', 'numeral', 'classifier', 'pronoun', 'preposition', 'multiword-expression', 'time-word', 'noun-of-locality']


def pos2map():
    pos2id = dict()
    for pos in reserve_pos_list:
        for order in ['B', 'I']:
            pos2id[order + '-' + pos] = len(pos2id) + 1
    pos2id['O'] = len(pos2id) + 1
    pos2id['[CLS]'] = len(pos2id) + 1
    pos2id['[SEP]'] = len(pos2id) + 1
    print(pos2id)
    with open(file_pos_pkl, 'wb+') as fw:
        pickle.dump(pos2id, fw)
    with open(file_pos_txt, 'w+', encoding='utf-8') as wf:
        for pos, _id in sorted(pos2id.items(), key=lambda x: x[1]):
            wf.write(pos + '\t' + str(_id) + '\n')


def get_pos_map():
    if not os.path.exists(file_pos_pkl):
        pos2map()
    pos2id = dict()
    with open(file_pos_pkl, 'rb') as fr:
        pos2id = pickle.load(fr)
    return pos2id


def main(input_file, output_file):
    pynlpir.open()
    fw = open(output_file, 'w+', encoding='utf-8')
    pos2id = get_pos_map()
    data = read_corpus(input_file)
    for _sent, _tags in data:
        sent = ''.join(_sent)
        result = pynlpir.segment(sent, pos_tagging=True, pos_names='parent', pos_english=True)
        # print(result)
        i = 0
        for _word, _speech in result:
            for j in range(len(_word)):
                char = _word[j]
                speech = ''
                if _speech is None or _speech not in reserve_pos_list:
                    speech = 'O'
                else:
                    speech = '-'.join(_speech.split(' '))
                    if j == 0:
                        speech = 'B-' + speech
                    else:
                        speech = 'I-' + speech
                if i >= len(_tags):
                    print(i, len(_sent), _sent)
                fw.write(char + ' ' + _tags[i] + ' ' + speech + '\n')
                i += 1
        fw.write('\n')
    fw.close()
    pynlpir.close()


if __name__ == '__main__':
    main('./../data/opera2/test.txt', './../data/opera2-pos/test.txt')
