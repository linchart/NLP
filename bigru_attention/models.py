# encoding: utf-8

"""

@author: linchart
@file: model.py
@version: 1.0
@time : 2019/1/20

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNClassifier(nn.Module):
    def __init__(self, nembedding, hidden_size, num_layer, dropout,
                 vocab_size, label_size, use_pretrain=False, embed_matrix=None, embed_freeze=True):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, nembedding)
        if use_pretrain is True:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embed_matrix).type(torch.FloatTensor),
                                                 requires_grad=not embed_freeze)

        self.gru = nn.GRU(input_size=nembedding,
                          hidden_size=hidden_size,
                          num_layers=num_layer,
                          dropout=dropout,
                          bidirectional=False)
        self.dense = nn.Linear(in_features=hidden_size,
                               out_features=label_size)

    def forward(self, sequences):
        padded_sentences, lengths = pad_packed_sequence(sequences, padding_value=int(0))
        embeds = self.embedding(padded_sentences)
        packed_embeds = pack_padded_sequence(embeds, lengths)
        out, _ = self.gru(packed_embeds)
        out, lengths = pad_packed_sequence(out, batch_first=False)
        lengths = [l - 1 for l in lengths]
        last_output = out[lengths, range(len(lengths))]
        logits = self.dense(last_output)
        output = F.log_softmax(logits, dim=1)
        return output


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.w = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state):

        alpha = F.softmax(self.w(hidden_state), dim=1)
        h_output = (hidden_state * alpha).sum(dim=0)

        return h_output


class Attentionclassifier(nn.Module):

    def __init__(self,
                 vocab_size,
                 emb_dim,
                 hidden_size,
                 num_layer,
                 dropout,
                 bidirectional,
                 label_size,
                 use_pretrain=True,
                 embed_matrix=None,
                 embed_freeze=False
                 ):
        super(Attentionclassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        if use_pretrain is True:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embed_matrix).type(torch.FloatTensor),
                                                 requires_grad=not embed_freeze)

        self.gru = nn.GRU(input_size=emb_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layer,
                          dropout=dropout,
                          bidirectional=bidirectional)

        self.attention = Attention(hidden_size * 2)

        self.dense = nn.Linear(in_features=hidden_size * 2,
                               out_features=label_size)



    def forward(self, sequences):
        padded_sentences, lengths = pad_packed_sequence(sequences, padding_value=int(0))
        embeds = self.embedding(padded_sentences)
        packed_embeds = pack_padded_sequence(embeds, lengths)
        out, _ = self.gru(packed_embeds)
        out, lengths = pad_packed_sequence(out, batch_first=False)

        output = self.attention(out)
        output = F.log_softmax(self.dense(output), dim=1)
        return output



class FinetuneModel1(nn.Module):

    def __init__(self,
                 vocab_size,
                 emb_dim,
                 hidden_size,
                 num_layer,
                 dropout,
                 bidirectional,
                 label_size,
                 hidden_size1,
                 use_pretrain=True,
                 embed_matrix=None,
                 embed_freeze=False
                 ):
        super(FinetuneModel1, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.hidden_size = hidden_size

        if use_pretrain is True:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embed_matrix).type(torch.FloatTensor),
                                                 requires_grad=not embed_freeze)

        self.gru = nn.GRU(input_size=emb_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layer,
                          dropout=dropout,
                          bidirectional=bidirectional)

        self.attention = Attention(hidden_size * 2)

        self.final = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size * 2, hidden_size1),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size1, label_size)
        )



    def forward(self, sequences):
        padded_sentences, lengths = pad_packed_sequence(sequences, padding_value=int(0))
        embeds = self.embedding(padded_sentences)
        packed_embeds = pack_padded_sequence(embeds, lengths)
        out, _ = self.gru(packed_embeds)
        out, lengths = pad_packed_sequence(out, batch_first=False)

        # 直接将out 放进attention 好像并不严谨
        output = self.attention(out)
        output = F.log_softmax(self.final(output), dim=1)
        return output



class FinetuneModel(nn.Module):

    def __init__(self, model, hidden_size1, class_size):
        super(FinetuneModel, self).__init__()
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.hidden_size = model.dense.in_features
        self.final = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, hidden_size1),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size1, class_size)
        )

    def forward(self, x):
        padded_sentences, lengths = pad_packed_sequence(x, padding_value=int(0))
        print("padded_sentences:{}".format(padded_sentences))
        x = self.model(padded_sentences)
        output = F.log_softmax(self.final(x), dim=1)
        return output
