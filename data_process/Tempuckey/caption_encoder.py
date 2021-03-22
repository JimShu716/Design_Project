import numpy as np
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax

import gensim.models.keyedvectors
from gensim.models import Word2Vec

"""
    Function to preprocess the raw caption text into one-hot encodings

    feature - raw caption feature
    word_dict - dictionary of words to construct one- hot

    Return: the processed caption feature
"""

def caption_to_one_hot(feature, word_vocab):
    dict_size = word_vocab.__len__()

    for i in range(len(feature)):
        for j in range(len(feature[i])):
            sentence = feature[i][j][1]

            sentence = sentence.lower()  # to lower case
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))  # remove all punctuations
            sentence_word = sentence.split()

            integer_encoded_sentence = []

            for word in sentence_word:
                word_integer = word_vocab.__call__(word)

                integer_encoded_sentence.append(word_integer)

            # print(integer_encoded_sentence)

            # ================ Initialize matrix for one hot encoding===========
            one_hot_sentence = []

            for idx in range(len(integer_encoded_sentence)):
                initial_arr = np.zeros(dict_size).tolist()
                initial_arr[integer_encoded_sentence[idx]] = 1.0
                one_hot_sentence.append(initial_arr)

            one_hot_sentence = np.array(one_hot_sentence)
            feature[i][j] = one_hot_sentence

    return feature


"""
Function to preprocess the raw caption text into one-hot encodings

feature - raw caption feature
word_dict - dictionary of words to construct one- hot

Return: the processed bow feature
"""


def caption_to_bow(feature, word_vocab):
    dict_size = word_vocab.__len__()
    feature_bow = []
    for i in range(len(feature)):
        sentence = ""
        for j in range(len(feature[i])):
            timestamps = feature[i][j][0]
            sentence += feature[i][j][1]
        # print("sentence is =",sentence)
        sentence = sentence.lower()  # to lower case
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))  # remove all punctuations
        sentence_word = sentence.split()

        integer_encoded_sentence = []

        for word in sentence_word:
            word_integer = word_vocab.__call__(word)
            if word_integer == -1:
                continue
            integer_encoded_sentence.append(word_integer)

        # print(integer_encoded_sentence)

        # ================ Initialize matrix for one hot encoding===========
        # one_hot_sentence = []
        one_hot_sentence = np.zeros(dict_size).tolist()
        for idx in range(len(integer_encoded_sentence)):
            one_hot_sentence[integer_encoded_sentence[idx]] = 1.0
            # one_hot_sentence.append(initial_arr)

        one_hot_sentence = np.array(one_hot_sentence)
        feature_bow.append(one_hot_sentence)

    return feature_bow


"""
Function to preprocess the data in txt file into word vocabulary

Return: the extracted word vocabulary
"""


def txt_to_vocabulary(file_path):
    word_vocab = ""

    # print("====== Start processing vocabulary ====")
    with open(file_path, 'rb') as reader:
        for line in reader:
            line = line.decode("utf-8")
            cap_id, caption = line.split(' ', 1)
            caption = caption.lower()  # all to lower case
            caption = caption.translate(str.maketrans('', '', string.punctuation))  # remove all punctuations
            word_vocab += ""
            word_vocab += caption

    vocab_result = word_vocab.split()

    # ========= Remove duplicates in the vocabulary ========
    vocab_set = set()
    final_vocab = []

    for word in vocab_result:
        if word not in vocab_set:
            vocab_set.add(word)
            final_vocab.append(word)

    return final_vocab


"""
Function to preprocess the word vocabulary into word dictionary for one-hot

Return: the word dictionary
"""


def vocab_to_dict(vocabulary):
    # print("====== Start constructing dictionary ====")
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(vocabulary)  # encode labels
    integer_encoded_list = integer_encoded.tolist()

    # ================ Construct a word dictionary===========
    word_dict = {}

    for key in vocabulary:
        for value in integer_encoded_list:
            word_dict[key] = value
            integer_encoded_list.remove(value)
            break
    # print("==== Dictionary Construction Completed =====")
    return (word_dict)


"""
Function to generate word embedding by word2vec

feature - feature with raw caption texts

Return: the feature with sentence embeddings
"""


def word2vec_embeddings(feature):
    # Load pretrained model
    model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True, unicode_errors='ignore')

    for i in range(len(feature)):
        for j in range(len(feature[i])):
            timestamps = feature[i][j][0]
            sentence = feature[i][j][1]

            sentence = sentence.lower()  # to lower case
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))  # remove all punctuations
            sentence_word = sentence.split()

            sentence_embeddings = []

            # ======== Generate word embeddings and sentence embeddings by pretrained word2vec
            for word in sentence_word:
                word_embeddings = model[word]
                sentence_embeddings.append(word_embeddings)

            feature[i][j] = (timestamps, sentence_embeddings)

            return feature


def compress_sentences(sentences):
    compressed_sentence = ''
    start_time = None
    end_time = None
    for sen in sentences:
        if type(sen) == tuple and len(sen) == 2:
            compressed_sentence += sen[1] + ' '
            if start_time is None:
                start_time = sen[0][0]
                end_time = sen[0][1]
            else:
                end_time = sen[0][1]
        elif type(sen) == str:
            compressed_sentence += sen + ' '
        else:
            raise RuntimeError
    compressed_sentence = compressed_sentence[:-1]
    return start_time, end_time, compressed_sentence


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, text_style):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.text_style = text_style

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return -1
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


print('1')