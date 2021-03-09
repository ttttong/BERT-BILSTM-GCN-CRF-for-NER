import re
import numpy as np

def create_graph_from_sentence_and_word_vectors(sentence, max_seq_length):
    from .nl import Parser

    if not isinstance(sentence, str):
        raise TypeError("String must be an argument")

    sentence = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r' \1_\2 ', sentence)

    parser = Parser(sentence, max_seq_length)

    A_fw = parser.execute()
    A_bw = parser.execute_backward()

    return A_fw, A_bw


tags = ["B-noun", "I-noun", "B-verb", "I-verb", "B-adjective", "I-adjective", "B-numeral", "I-numeral", "B-classifier", "I-classifier", "B-pronoun", "I-pronoun", "B-preposition", "I-preposition", "B-multiword-expression", "I-multiword-expression", "B-time-word", "I-time-word", "B-noun-of-locality", "I-noun-of-locality", "O", "[CLS]", "[SEP]"]
classes = ["X", "B-剧种", "I-剧种", "B-剧目", "I-剧目", "B-乐器", "I-乐器", "B-地点", "I-地点", "B-唱腔曲牌", "I-唱腔曲牌", "B-脚色行当", "I-脚色行当", "O", "[CLS]", "[SEP]"]


def get_all_sentences(filename):
    file = open(filename, encoding='utf-8')
    sentences = []
    items = []
    for line in file.readlines():
        elements = line.split()
        if len(elements) == 0:
            if items != []:
                sentences.append(items)
                items = []
                continue
        word = elements[0]
        entity = elements[1]
        tag = elements[2]
        items.append((word, tag, entity))
    sentences.append(items)
    return sentences


def decide_entity(string, prior_entity):
    if string == '*)':
        return prior_entity, ''
    if string == '*':
        return prior_entity, prior_entity
    entity = ''
    for item in classes:
        if string.find(item) != -1:
            entity = item
    prior_entity = ''
    if string.find(')') == -1:
        prior_entity = entity
    return entity, prior_entity


def get_clean_word_vector(word):

    from spacy.lang.zh import Chinese
    parser = Chinese()
    default_vector = parser('entity')[0].vector

    parsed = parser(word)
    try:
        vector = parsed[0].vector
        if vector_is_empty(vector):
            vector = default_vector
    except:
        vector = default_vector
    return np.array(vector, dtype=np.float64)


def vector_is_empty(vector):
    to_throw = 0
    for item in vector:
        if item == 0.0:
            to_throw += 1
    if to_throw == len(vector):
        return True
    return False


def get_class_vector(class_name):
    vector = [0.] * (len(classes) + 1)
    index = len(classes)
    try:
        index = classes.index(class_name)
    except:
        pass
    vector[index] = 1.
    return vector


def get_tagging_vector(tag):
    vector = [0.] * (len(tags) + 1)
    index = len(tags)
    try:
        index = tags.index(tag)
    except:
        pass
    vector[index] = 1.
    return vector


def get_data_from_sentences(sentences):
    all_data = []
    for sentence in sentences:
        word_data = []
        class_data = []
        tag_data = []
        words = []
        for word, tag, entity in sentence:
            words.append(word)

            word_vector = get_clean_word_vector(word)
            word_data.append(word_vector)

            tag_vector = get_tagging_vector(tag)
            tag_data.append(tag_vector)

            class_vector = get_class_vector(entity)
            class_data.append(class_vector)
        all_data.append((words, word_data, tag_data, class_data))
    return all_data


def create_full_sentence(words):
    import re
    sentence = ''.join(words)
    return sentence


def get_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def bin_data_into_buckets(data, batch_size):
    buckets = []
    size_to_data_dict = {}
    for item in data:
        sequence = item[0]
        length = len(sequence)
        try:
            size_to_data_dict[length].append(item)
        except:
            size_to_data_dict[length] = [item]
    for key in size_to_data_dict.keys():
        data = size_to_data_dict[key]
        chunks = get_chunks(data, batch_size)
        for chunk in chunks:
            buckets.append(chunk)
    return buckets