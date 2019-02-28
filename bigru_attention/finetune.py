# encoding: utf-8

"""

@author: linchart
@file: finetune.py
@version: 1.0
@time : 2019/1/21

"""

import config
import logging
import models
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data
import main
import predict



def finetune():
    vocab = data.Vocabulary()
    data.build_vocab(vocab, config.vector_file)  # build vocabulary

    train_data = data.Sentiment(config.finetune_train_file, vocab)

    train_dataloader = DataLoader(train_data,
                                  batch_size=config.TRAIN_BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=data.collate_fn)

    valid_data = data.Sentiment(config.finetune_valid_file, vocab)

    valid_dataloader = DataLoader(valid_data,
                                 batch_size=config.TRAIN_BATCH_SIZE,
                                 shuffle=True,
                                 collate_fn=data.collate_fn)

    test_data = data.Sentiment(config.finetune_test_file, vocab)

    test_dataloader = DataLoader(test_data,
                                 batch_size=config.TRAIN_BATCH_SIZE,
                                 shuffle=True,
                                 collate_fn=data.collate_fn)



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


    # 将pretrained_dict里不属于model_dict的键剔除掉

    pretrained_dict = dict()

    for k, v in pretrained_model.items():
        if k == 'state_dict':
            for kk, vv in v.items():
                if kk in model_dict:
                    pretrained_dict[kk] = vv



    # 更新现有的model_dict
    model_dict.update(pretrained_dict)

    # 加载实际需要的model_dict
    classifier.load_state_dict(model_dict)


    # 固定网络参数，不更新
    for param in classifier.parameters():
        param.requires_grad = False

    # 将最后final层的参数设置可以更新
    for param in classifier.final.parameters():
        param.requires_grad = True


    # new_model = models.FinetuneModel(classifier, hidden_size1=128, class_size=2)
    # print(new_model)

    criterion = nn.NLLLoss()
    # optimizer = torch.optim.Adam(classifier.parameters())
    # optimizer = torch.optim.RMSprop(classifier.parameters(), lr=0.001, alpha=0.9, momentum=0.2)
    optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, classifier.parameters()),
                                     lr=0.01, rho=0.9, eps=1e-06, weight_decay=0)
    # optimizer = torch.optim.RMSprop(classifier.parameters())

    best_f1 = 0

    for epoch in range(config.finetune_epochs):

        # lr update
        # adjust_learning_rate(optimizer, epoch)
        # 测试不同优化器的学习率是否是自适应的
        for param_group in optimizer.param_groups:
            print("here lr :{}".format(param_group['lr']))

        logging.info("epoch {0:04d}".format(epoch))
        main.train(train_dataloader, classifier, criterion, optimizer, epoch, config.finetune_batch_size, config.silent)
        test_f1 , val_loss= main.test(valid_dataloader, classifier, criterion, epoch, config.finetune_batch_size, config.silent)

        is_best = test_f1 > best_f1  # True or False
        best_f1 = max(test_f1, best_f1)

        logging.info("best f1 is {}".format(best_f1))
        main.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': classifier.state_dict(),
            'acc': test_f1,
            'best_acc': best_f1,
            'optimizer': optimizer.state_dict(),
        },
            is_best,
            checkpoint='../output/',save_file = 'finetune_model_best.pth.tar')

    predict.predict(classifier, test_dataloader, config.silent)


if __name__ == '__main__':
    finetune()
