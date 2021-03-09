import os
import pickle
import pynlpir
import re
from processor.data import read_corpus


# file_name = 'train'
# file_input = '../data/MSRA/{}.txt'.format(file_name)
# file_output = '../data/MSRA-pos/{}.txt'.format(file_name)
# file_pos_pkl = '../data/MSRA-pos/pos2id.pkl'
# file_pos_txt = '../data/MSRA-pos/pos2id.txt'

# file_name = 'test'
# file_input = '../data/MSRA/{}.txt'.format(file_name)
# file_output = '../data/MSRA-pos/{}.txt'.format(file_name)
file_pos_pkl = '../data/opera2-all/mark2id.pkl'
file_pos_txt = '../data/opera2-all/mark2id.txt'

#
reserve_pos_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']


def pos2map():
    pos2id = dict()
    for pos in reserve_pos_list:
        pos2id[pos] = len(pos2id) + 1
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
    a=''
    fr = open(input_file, 'r', encoding='utf-8')
    pos2id = get_pos_map()
    for i in fr.readlines():
        if i == None:
            a = a+'\n'
        elif i == '\n':
            a = a+'\n'
        elif i[0] == '省':
            i = i.replace('\n', ' a\n')
            a = a+i
        elif i[0] == '市':
            i = i.replace('\n', ' b\n')
            a = a+i
        elif i[0] == '戏':
            i = i.replace('\n', ' c\n')
            a = a+i
        elif i[0] == '剧':
            i = i.replace('\n', ' d\n')
            a = a+i
        elif i[0] == '《':
            i = i.replace('\n', ' e\n')
            a = a+i
        elif i[0] == '》':
            i = i.replace('\n', ' f\n')
            a = a+i
        elif i[0] == '“':
            i = i.replace('\n', ' g\n')
            a = a+i
        elif i[0] == '【':
            i = i.replace('\n', ' h\n')
            a = a+i
        elif i[0] == '】':
            i = i.replace('\n', ' i\n')
            a = a+i
        elif i[0] == '[':
            i = i.replace('\n', ' j\n')
            a = a+i
        elif i[0] == ']':
            i = i.replace('\n', ' k\n')
            a = a+i
        else:
            i = i.replace('\n', ' O\n')
            a = a+i
    fr = open(output_file, 'w+', encoding='utf-8')
    fr.write(a)

if __name__ == '__main__':
    main('./../data/opera2-pos/test.txt', './../data/opera2-all/test.txt')
