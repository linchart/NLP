# encoding: utf-8

"""

@author: linchart
@file: prob2file.py
@version: 1.0
@time : 2018/12/2

"""
import os

from hmm_prob import *

def write_prob_start(start_prob, file):
    with open(file, 'w') as f:
        f.write('P = {')
        for key, value in start_prob.items():
            f.write('\'%s\':%s,\n' % (key, value))
        f.write('}')


def write_prob_trans(prob_trans, file):
    with open(file, 'w') as f:
        f.write('P = {')
        for key, value in prob_trans.items():
            f.write('\'%s\': {' % key)
            for key_, value_ in value.items():
                f.write('\'%s\': %s,' % (key_, value_))
            f.write('},\n')
        f.write('}')


def write_prob_emit(prob_emit, file):
    with open(file, 'w', encoding='utf-8') as f:
        f.write('P = {')
        for key, value in prob_emit.items():
            f.write('\'%s\': {' % key)
            for key_, value_ in value.items():
                f.write('\'%s\': %s,\n' % (key_, value_))
                f.write('        ')
            f.write('},\n    ')
        f.write('}')


if __name__ == '__main__':
    file_new = r"E:\work\NLP研究应用\分词\input\data.txt"
    hmm_init = Initial()
    start_prob_init, trans_prob_init, emit_prob_init = hmm_init.init_prob(file_new)

    output_file = r'E:\work\NLP研究应用\分词\output'
    write_prob_start(start_prob_init, os.path.join(output_file, 'prob_start.py'))
    write_prob_trans(trans_prob_init, os.path.join(output_file, 'prob_trans.py'))
    write_prob_emit(emit_prob_init, os.path.join(output_file, 'prob_emit.py'))
