# encoding: utf-8

"""

@author: linchart
@file: data.py
@version: 1.0
@time : 2019/1/20

"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import csv
from torch.utils.data.sampler import SequentialSampler
from torch.autograd import Variable
import gensim

import config




class Sentiment(Dataset):

    def __init__(self, file, vocab):
        data = pd.read_csv(file)
        # data['comment'] = data.apply(lambda x: x.str.split(' '), axis=1)
        data = data[data['comment'].str.len() > 4]
        data['comment'] = data.apply(lambda x: x.str.split(' '), axis=1)
        self.dataset = data
        self.comment = self.dataset['comment'].values
        self.label = self.dataset['label'].values

        self.vocab = vocab

    def __getitem__(self, index):
        data_idx = {'comment': [], 'label': self.label[index]}
        try:
            for i, word in enumerate(self.comment[index]):
                data_idx['comment'].append(self.vocab.get_idx(word))
        except:
            print("typeerror: {}".format(self.comment[index]))

        return data_idx  # return index rather than word

    def __len__(self):
        return len(self.dataset)



def collate_fn(batch):  # rewrite collate_fn to form a mini-batch
    lengths = np.array([len(data['comment']) for data in batch])
    sorted_index = np.argsort(-lengths)
    lengths = lengths[sorted_index]  # descend order

    max_length = lengths[0]
    batch_size = len(batch)

    sentence_tensor = torch.LongTensor(batch_size, int(max_length)).zero_()

    for i, index in enumerate(sorted_index):
        sentence_tensor[i][:lengths[i]] = torch.LongTensor(batch[index]['comment'][:max_length])

    sentiments = torch.LongTensor([int(batch[i]['label']) for i in sorted_index])

    packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor.t().to(config.device),
                                                               lengths)  # remember to transpose
    sentiments = sentiments.to(config.device)

    return {'comment': packed_sequences, 'label': sentiments}



def build_vocab(vocab, vector_file):
    word_vector = gensim.models.KeyedVectors.load_word2vec_format(
        fname=vector_file, binary=True)
    words = word_vector.vocab
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')
    unk_vector = np.random.uniform(-0.25, 0.25, size=config.DIM)  # unk random initial
    vocab.vector = np.zeros((config.MAX_VOCAB_SIZE, config.DIM), dtype=np.float)
    vocab.vector[1][:] = unk_vector
    for word in words:
        vocab.vector[vocab.n_words][:] = word_vector[word]
        vocab.add_word(word)
        if vocab.n_words == config.MAX_VOCAB_SIZE:
            break


class Vocabulary():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.n_words = 0
        self.vector = ""

    def get_idx(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def get_word(self, idx):
        if idx >= self.n_words:
            print("index out of range")
            return None
        else:
            return self.idx2word[idx]

    def add_word(self, word):
        self.word2idx[word] = self.n_words
        self.idx2word[self.n_words] = word
        self.n_words += 1
