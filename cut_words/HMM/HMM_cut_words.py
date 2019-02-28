# encoding: utf-8

"""

@author: linchart
@file: HMM_cut_words.py
@version: 1.0
@time : 2018/12/03

"""

import re

from viterbi import *
from prob_start import P as start_P
from prob_trans import P as trans_P
from prob_emit import P as emit_P

status = 'BEMS'
re_han = re.compile("([\u4E00-\u9FD5]+)")
re_skip = re.compile("(\d+\.\d+|[a-zA-Z0-9]+)")


def __cut(sen):
    prob, pos_list = viterbi(sen, status, start_P, trans_P, emit_P, end_status='ES')
    flag = 0
    for num, pos in enumerate(pos_list):
        if pos == 'E':
            word = sen[flag: num + 1]
            flag = num + 1
            yield word

        elif pos == 'S':
            word = sen[flag: num + 1]
            flag = num + 1
            yield word

def cut(sen):
    blocks = re_han.split(sen)
    for blk in blocks:
        if re_han.match(blk):
            for word in __cut(blk):
                yield word
        else:
            print("blk is {}".format(blk))
            tmp = re_skip.split(blk)
            for x in tmp:
                if x:
                    yield x



if __name__ == '__main__':
    sentence = '瓦西里斯的船只中有４０％驶向远东，每个月几乎都有两三条船停靠中国港口。'
    print(list(cut(sentence)))
