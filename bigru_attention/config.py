# encoding: utf-8

"""

@author: linchart
@file: config.py
@version: 1.0
@time : 2019/1/20

"""

import torch

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

TRAIN_BATCH_SIZE = 32
DIM = 200
HIDDEN_SIZE = 128
NUM_LAYER = 1
drop_out = 0.3
epochs = 30
silent = False
label_class = 2
bidirectional = True
LR = 0.001

finetune_epochs=500
finetune_batch_size = 20


train_file = '../output/train_char.csv'
test_file = '../output/test_char1.csv'
predict_file = '../input/other_test_data_deal.csv'
# predict_file = '../input/finetune_test_data.csv'
finetune_train_file = '../input/finetune_train_data.csv'
finetune_valid_file = '../input/finetune_validation_data.csv'
finetune_test_file = '../input/finetune_test_data.csv'



vector_file = '../output/chars.vector'

# model_path = '../output/model_best.pth.tar'
model_path = '../output/finetune_model_best.pth.tar'
