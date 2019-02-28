# encoding: utf-8

"""

@author: linchart
@file: viterbi.py
@version: 1.0
@time : 2018/11/30

"""


PrevStatus = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}

MIN_FLOAT = -3.14e100

def viterbi(obs, status, prob_start, prob_trans, prob_emit, end_status='ES'):
    """
    基于动态规划(viterbi)求解隐马尔科夫预测问题
    :param obs: 观测序列
    :param status: 状态集合 string
    :param prob_start: 初始状态分布
    :param prob_trans: 状态转移概率矩阵
    :param prob_emit: 发射概率矩阵
    :param end_status: 指定观测结束位置对应的可能状态
    :return: 最有可能的状态序列，及对应的概率
    """
    v = [{}]
    path = {}
    for t, word in enumerate(obs):
        if t == 0:
            for statu in status:
                v[0][statu] = prob_start[statu] + prob_emit[statu].get(word, MIN_FLOAT)
                path[statu] = [statu]
        else:
            prob_tmp = v[0]
            v[0] = {}
            newpath = {}
            for statu in status:
                prob_list = []
                for y in PrevStatus[statu]:
                    prob = prob_tmp[y] + prob_trans[y].get(statu) + prob_emit[statu].get(word, MIN_FLOAT)
                    prob_list.append((prob, y))
                prob_max, state = max(prob_list)
                v[0][statu] = prob_max
                newpath[statu] = path[state] + [statu]
            path = newpath
        # print("now word is {}".format(word))
        # print(v[0])
    (prob, state) = max((v[0][y], y) for y in end_status)
    return prob, path[state]
