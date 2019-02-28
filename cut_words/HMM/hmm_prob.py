# encoding: utf-8

"""

@author: linchart
@file: hmm_prob.py
@version: 1.0
@time : 2018/12/2

"""

import re
import math
from collections import Counter

init_trans_freq_dict = {'B': {}, 'E': {}, 'M': {}, 'S': {}}


class Initial:

    def __init__(self):
        self.spliter = re.compile('[、，。？；：,\.\?:]')
        self.s = re.compile('^/s')
        self.remain = re.compile('/s|/b|/m|/e')
        self.start = {'/b': 0, '/s': 0}
        self.status = 'BEMS'
        self.prob_start = {'B': -3.14e+100, 'E': -3.14e+100, 'M': -3.14e+100, 'S': -3.14e+100}

    def __freq(self, file):
        trans_list = []
        emit_count = {'/b': {}, '/e': {}, '/m': {}, '/s': {}}
        with open(file, encoding='utf-8', mode='r') as f:

            for line in f:
                con_for_emit = line.rstrip().split(' ')
                # 计算发射矩阵观测频率
                for word in con_for_emit:
                    word_ = str(word[0])
                    statu = word[1:]
                    try:
                        emit_count[statu][word_] = emit_count[statu].get(word_, 0) + 1
                    except KeyError:
                        continue

                split_content = self.spliter.split(line)
                for content in split_content:
                    con = re.sub(self.s, "", content)
                    con_ = re.findall(self.remain, con)

                    if len(con_) <= 1:
                        continue
                    # 计算初始频率
                    try:
                        self.start[con_[0]] += 1
                    except KeyError:
                        print("wrong key : {}".format(con_[0]))
                    except IndexError:
                        pass
                    con_len = len(con_)
                    # 统计状态转移频率
                    for num in range(con_len - 1):
                        trans_obs = ''.join(con_[num: num + 2])
                        trans_list.append(trans_obs)
        return emit_count, trans_list

    def init_prob_start(self):
        init_freq = sum(list(self.start.values()))
        for key in self.start:
            self.prob_start[key[1].upper()] = math.log(self.start.get(key) / init_freq)

    def init_prob_trans(self, trans):
        trans_count = Counter(trans)
        trans_freq_dict = {}
        for statu in trans_count:
            init_trans_freq_dict[statu[1].upper()][statu[3].upper()] = trans_count[statu]

        for statu, stov in init_trans_freq_dict.items():
            trans_freq_dict[statu] = {}
            for key in stov:
                trans_freq_dict[statu][key] = math.log(stov[key] / sum(list(stov.values())))
        return trans_freq_dict

    def init_prob_emit(self, emit):
        emit_freq_dict = {}
        for key in emit:
            emit_statu = key[1].upper()
            emit_freq_dict[emit_statu] = {}
            total_freq = sum(list(emit[key].values()))
            word_dict = emit[key]
            for em in word_dict:
                emit_freq_dict[emit_statu][em] = math.log(word_dict[em] / total_freq)
        return emit_freq_dict

    def init_prob(self, file):

        emit_freq, trans_freq = self.__freq(file)

        # 初始状态概率
        self.init_prob_start()
        start_prob_dict = self.prob_start

        # 状态转移概率
        trans_prob_dict = self.init_prob_trans(trans_freq)

        # 发射概率
        emit_prob_dict = self.init_prob_emit(emit_freq)

        return start_prob_dict, trans_prob_dict, emit_prob_dict


if __name__ == '__main__':
    file_1 = r"E:\work\NLP研究应用\分词\input\data.txt"
    hmm_init = Initial()
    start_prob, trans_prob, emit_prob = hmm_init.init_prob(file_1)


#!/usr/bin/python
import json

data = [ { 'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5 } ]

json = json.dumps(data)

