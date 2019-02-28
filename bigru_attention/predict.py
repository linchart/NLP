
# encoding: utf-8

"""

@author: linchart
@file: predict.py
@version: 1.0
@time : 2019/1/21

"""

from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data
import config
import models


def tfpn(predition, target):

    tp = ((predition == 1) & (target.view_as(predition) == 1)).sum().item()

    fn = ((predition == 0) & (target.view_as(predition) == 1)).sum().item()

    fp = ((predition == 1) & (target.view_as(predition) == 0)).sum().item()

    tn = ((predition == 0) & (target.view_as(predition) == 0)).sum().item()
    return tp, fn, fp, tn


def predict(model, predict_data, silent):

    model.eval()

    data_iterator = tqdm(predict_data,
                         desc='predict',
                         leave=True,
                         unit='batch',
                         disable=silent)

    correct = 0

    tp_list = []
    fn_list = []
    fp_list = []
    tn_list = []

    for index, collate_data in enumerate(data_iterator):

        inputs = collate_data['comment']
        targets = collate_data['label']

        outputs = model(inputs)
        pred = outputs.max(1, keepdim=True)[1]

        correct += pred.eq(targets.view_as(pred)).sum().item()

        tp, fn, fp, tn = tfpn(pred, targets)
        tp_list.append(tp)
        fn_list.append(fn)
        fp_list.append(fp)
        tn_list.append(tn)

    precision = sum(tp_list) / (sum(tp_list) + sum(fp_list))
    recall = sum(tp_list) / (sum(tp_list) + sum(fn_list))
    f1_score = 2 * (precision * recall) / (precision + recall)

    acc = 100 * correct / len(predict_data.dataset)

    confusion_matrix = pd.DataFrame([[sum(tn_list), sum(fp_list)],[sum(fn_list), sum(tp_list)]],
                                    index=['real 0', 'real 1'],
                                    columns=['predict 0', 'predict 1'])

    print("precision：{precision} ，acc: {acc}, \nf1_score:{f1_score}".format(precision=precision, acc=acc, f1_score=f1_score))
    print("confusion_matrix:\n {}".format(confusion_matrix))

    return f1_score


def main():

    vocab = data.Vocabulary()
    data.build_vocab(vocab, config.vector_file)  # build vocabulary

    # classifier = models.Attentionclassifier(vocab_size=vocab.n_words,
    #                                         emb_dim=config.DIM,
    #                                         hidden_size=config.HIDDEN_SIZE,
    #                                         num_layer=config.NUM_LAYER,
    #                                         dropout=config.drop_out,
    #                                         bidirectional=config.bidirectional,
    #                                         label_size=config.label_class,
    #                                         use_pretrain=True,
    #                                         embed_matrix=vocab.vector,
    #                                         embed_freeze=False).to(config.device)

    classifier = models.FinetuneModel1(vocab_size=vocab.n_words,
                                       emb_dim=config.DIM,
                                       hidden_size=config.HIDDEN_SIZE,
                                       num_layer=config.NUM_LAYER,
                                       dropout=config.drop_out,
                                       bidirectional=config.bidirectional,
                                       label_size=config.label_class,
                                       hidden_size1=128,
                                       use_pretrain=True,
                                       embed_matrix=vocab.vector,
                                       embed_freeze=False).to(config.device)



    model_dict = classifier.state_dict()

    pretrained_model = torch.load(config.model_path)


    pretrained_dict = dict()

    for k, v in pretrained_model.items():
        if k == 'state_dict':
            for kk, vv in v.items():
                if kk in model_dict:
                    pretrained_dict[kk] = vv


    # # 更新现有的model_dict
    model_dict.update(pretrained_dict)

    # 加载实际需要的model_dict
    classifier.load_state_dict(model_dict)
    # classifier.eval()
    test_data = data.Sentiment(config.predict_file, vocab)
    test_dataloader = DataLoader(test_data,
                                 batch_size=config.TRAIN_BATCH_SIZE,
                                 shuffle=True,
                                 collate_fn=data.collate_fn)
    predict(classifier, test_dataloader, config.silent)

if __name__ == '__main__':
    main()
