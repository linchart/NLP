# encoding: utf-8

"""

@author: linchart
@file: main.py
@version: 1.0
@time : 2019/1/20

"""

import shutil
import time
import os
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data
import config
import models


logging.captureWarnings(True)

format_time = time.strftime('%Y%m%d', time.localtime())
log_file = '../log/sentiment_model_{}'.format(format_time)
logging.basicConfig(filename='{}'.format(log_file),
                    format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%Y-%M-%d %H:%M:%S',
                        level=logging.DEBUG, filemode='w')

def train(trainloader, model, criterion, optimizer, epoch, batch_size, silent):

    model.train()

    data_iterator = tqdm(trainloader,
                         desc='train',
                         leave=True,
                         unit='batch',
                         postfix={'epo': epoch, 'loss': '%.6f' % 0.0, 'acc': '%.6f' % 0.0},
                         disable=silent)

    correct = 0
    total_loss = 0

    for index, collate_data in enumerate(data_iterator):

        inputs = collate_data['comment'].to(config.device)
        targets = collate_data['label']

        optimizer.zero_grad()

        outputs = model(inputs)
        pred = outputs.max(1, keepdim=True)[1]
        pred = pred.to(config.device)
        targets = targets.squeeze().to(config.device)
        batch_correct_ = pred.eq(targets.view_as(pred)).sum().item()
        batch_correct = batch_correct_ / batch_size
        correct += batch_correct_

        batch_loss = criterion(outputs, targets)
        loss_value = float(batch_loss.item())
        total_loss += loss_value


        batch_loss.backward()
        optimizer.step(closure=None)

        data_iterator.set_postfix(epo=epoch,
                                  loss='%.6f' % loss_value,
                                  acc='%.6f' % batch_correct)

    acc = 100 * correct / len(trainloader.dataset)
    # 计算每个batch_size 的loss
    loss = total_loss / (len(trainloader.dataset)/batch_size)

    logging.info("     loss: {loss} , acc: {acc}".format(epoch=epoch, loss=loss, acc=acc))


def tfpn(predition, target):

    tp = ((predition == 1) & (target.view_as(predition) == 1)).sum().item()

    fn = ((predition == 0) & (target.view_as(predition) == 1)).sum().item()

    fp = ((predition == 1) & (target.view_as(predition) == 0)).sum().item()

    tn = ((predition == 0) & (target.view_as(predition) == 0)).sum().item()
    return tp, fn, fp, tn


def test(testloader, model, criterion, epochs, batch_size, silent):
    model.eval()

    data_iterator = tqdm(testloader,
                         desc='test',
                         leave=True,
                         unit='batch',
                         postfix={'epo': epochs},
                         disable=silent,
                         )

    total_loss = 0
    correct = 0

    tp_list = []
    fn_list = []
    fp_list = []
    tn_list = []

    for index, collate_data in enumerate(data_iterator):
        inputs = collate_data['comment']
        targets = collate_data['label']

        batch_size = targets.size()[0]
        outputs = model(inputs)
        pred = outputs.max(1, keepdim=True)[1]
        batch_loss = criterion(outputs, targets)
        loss_value = float(batch_loss.item())
        total_loss += loss_value

        correct += pred.eq(targets.view_as(pred)).sum().item()

        tp, fn, fp, tn = tfpn(pred, targets)
        tp_list.append(tp)
        fn_list.append(fn)
        fp_list.append(fp)
        tn_list.append(tn)
    # 待修正
    try:
        precision = sum(tp_list) / (sum(tp_list) + sum(fp_list))
    except ZeroDivisionError:
        precision = 0
    try:
        recall = sum(tp_list) / (sum(tp_list) + sum(fn_list))
    except ZeroDivisionError:
        recall = 0
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0

    acc = 100 * correct / len(testloader.dataset)
    # 计算每个batch_size 的loss
    loss = total_loss / (len(testloader.dataset)/batch_size)

    logging.info("     val_loss: {loss} , val_acc: {acc}, val_f1_score: {f1_score}".format(epoch=epochs,
                                                                                           loss=loss,
                                                                                           acc=acc,
                                                                                           f1_score=f1_score)
                 )
    return f1_score, loss


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar',
                    save_file = 'model_best.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, save_file))


def adjust_learning_rate(optimizer, epoch):
    "learning rate decayed by 10 every 10 epochs"
    lr = config.LR * (0.1 ** (epoch // 5))
    print("now lr is {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adam_optimizers(parameters):
    optimizer = torch.optim.Adam(parameters, lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, mode='min', verbose=True)
    return optimizer, scheduler


def main():

    best_f1 = 0
    print(config.device)

    vocab = data.Vocabulary()
    data.build_vocab(vocab, config.vector_file)  # build vocabulary

    train_data = data.Sentiment(config.train_file, vocab)

    train_dataloader = DataLoader(train_data,
                                  batch_size=config.TRAIN_BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=data.collate_fn)

    test_data = data.Sentiment(config.test_file, vocab)

    test_dataloader = DataLoader(test_data,
                                 batch_size=config.TRAIN_BATCH_SIZE,
                                 shuffle=True,
                                 collate_fn=data.collate_fn)

    # classifier = models.RNNClassifier(nembedding=config.DIM,
    #                                   hidden_size=config.HIDDEN_SIZE,
    #                                   num_layer=config.NUM_LAYER,
    #                                   dropout=config.drop_out,
    #                                   vocab_size=vocab.n_words,
    #                                   use_pretrain=True,
    #                                   embed_matrix=vocab.vector,
    #                                   embed_freeze=False,
    #                                   label_size=config.label_class).to(config.device)

    classifier = models.Attentionclassifier(vocab_size=vocab.n_words,
                                            emb_dim=config.DIM,
                                            hidden_size=config.HIDDEN_SIZE,
                                            num_layer=config.NUM_LAYER,
                                            dropout=config.drop_out,
                                            bidirectional=config.bidirectional,
                                            label_size=config.label_class,
                                            use_pretrain=True,
                                            embed_matrix=vocab.vector,
                                            embed_freeze=False).to(config.device)

    criterion = nn.NLLLoss()
    # optimizer = torch.optim.Adam(classifier.parameters())
    optimizer = torch.optim.RMSprop(classifier.parameters(), lr=config.LR, alpha=0.9, momentum=0.2)
    # optimizer = torch.optim.RMSprop(classifier.parameters())

    # optimizer, scheduler = adam_optimizers(classifier.parameters())

    # optimizer = torch.optim.Adadelta(classifier.parameters(), lr=config.LR, rho=0.9, eps=1e-06, weight_decay=0)

    for epoch in range(config.epochs):

        # lr update
        adjust_learning_rate(optimizer, epoch)
        # 测试不同优化器的学习率是否是自适应的
        # for param_group in optimizer.param_groups:
        #     print("here lr :{}".format(param_group['lr']))

        logging.info("epoch {0:04d}".format(epoch))
        train(train_dataloader, classifier, criterion, optimizer, epoch, config.TRAIN_BATCH_SIZE, config.silent)
        test_f1, val_loss = test(test_dataloader, classifier, criterion, epoch, config.TRAIN_BATCH_SIZE, config.silent)

        # scheduler.step(val_loss)

        is_best = test_f1 > best_f1  # True or False
        best_f1 = max(test_f1, best_f1)

        logging.info("best f1 is {}".format(best_f1))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': classifier.state_dict(),
            'acc': test_f1,
            'best_acc': best_f1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint='../output/')


if __name__ == '__main__':
    main()
