# encoding: utf-8

"""

@author: linchart
@file: discovery.py
@version: 1.0
@time : 2018/11/22

"""
import time
import logging
import argparse
import math
import pickle
import pandas as pd
from itertools import chain

from tire_tree import Tire


format_time = time.strftime('%Y%m%d', time.localtime())
logging.basicConfig(filename='../log/baseline_{}.log'.format(format_time),
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    filemode='w')
logger = logging.getLogger(__name__)


def n_gram(string, n):
    """
    获取一个字符串的n gram 词组合，词数大于等于2
    :param string:
    :param n:
    :return:
    """
    string_len = len(string)
    words = [list(string)]
    for gram in range(2, n+1):
        words.append([string[num: num + gram] for num in range(string_len - gram + 1)])
    n_gram_words = chain(*words)
    return n_gram_words


def word_split(word):
    """
    将词语拆分成其组合词，比如 炸薯条 可以拆分成 [炸薯, 条]、[炸, 薯条]
    :param word: 要拆分的词
    :return: 组合词
    """
    return ([word[: n], word[n:]] for n in range(1, len(word)))


# 优化的话，看能否弄成矩阵的形式？
# 计算 文本凝固度
def mutualinfo(words, tree):
    """
    计算每个词的互信息，也就是词文本凝固度，为防止切分方法高估该词的凝合程度，取互信息较小值作为
    该词的凝固度
    :param words: 所有词集合
    :param tree: Tire 树
    :return: 返回一个包含每个词对应凝固度的词典
    """
    word2mutual = {}
    for word in words:
        groups = word_split(word)
        # 词 word 的出现次数
        word_counts = tree.search(word)[1]
        mu_list = []

        for group in groups:
            # 如炸薯条，会分别计算[炸薯, 条]、[炸, 薯条] 的凝固度
            # p(薯条| 炸) = 炸薯条 出现次数 / 炸 出现的次数
            # p(薯条) = 薯条 出现次数 / 文档词 总数量
            # 计算 I(条, 炸薯) =  p(薯条| 炸) / p(薯条)
            prob_sub_y_x = word_counts / tree.search(group[0])[1]
            prob_sub_y = tree.search(group[1])[1] / tree.root.count
            mu_ = math.log(prob_sub_y_x / prob_sub_y)
            mu_list.append(mu_)

        min_mu = min(mu_list)
        word2mutual[word] = min_mu
    return word2mutual


# 找到一个词的左邻 词 和 右邻 词
def calc_entropy(words, tree):
    """
    计算 词的 左右信息熵
    :param words: 所有词的集合
    :param tree: Tire 树
    :return:
    """
    word2entropy = {}

    def entropy(sample, total_sample):
        """entropy"""
        return -sample / total_sample * math.log(sample / total_sample)

    for word in words:
        nodes = tree.fetch(word)
        total = nodes.count
        total_entropy = sum([entropy(value.count, total)
                             for value in nodes.tree.values()])
        word2entropy[word] = total_entropy

    return word2entropy

#
# def location_prob(words, fwtree, bwtree):
#     """
#     位置成词概率，效果并不好， 参考文档：http://www.docin.com/p-744511073.html
#     only when len(words) == 3
#     :param words:
#     :param fwtree:
#     :param bwtree:
#     :return:
#     """
#     k_group = list(words)
#     # fnodes1 = fwtree.fetch(word[0])
#     fnodes2 = fwtree.fetch(k_group[1])
#     bnodes2 = bwtree.fetch(k_group[1])
#     # bnodes3 = bwtree.fetch(word[2])
#
#     # type =1
#     # pwp1 = sum(value.count for value in fnodes1.tree.values()) / fnodes1.count
#     pwp21 = sum(value.count for value in fnodes2.tree.values()) / fnodes2.count
#     # pwp3 = sum(value.count for value in bnodes3.tree.values()) / bnodes3.count
#
#     # type =2
#     pwp22 = sum(value.count for value in bnodes2.tree.values()) / bnodes2.count
#
#     return max(math.log(pwp21), math.log(pwp22))


class _WordDiscovery:

    def __init__(self):
        self.fw_ngram = Tire()
        self.bw_ngram = Tire()
        self.puncs = ['【', '】', ')', '(', '、', '，', '“', '”',
                      '。', '《', '》', ' ', '-', '！', '？', '.',
                      '\'', '[', ']', '：', '/', '.', '"', '\u3000',
                      '’', '．', ',', '…', '?', ';', '·', '%', '（',
                      '#', '）', '；', '>', '<', '$', ' ', ' ', '\ufeff',
                      '*', '——']

    def preprocess(self, text, max_size):

        for punc in self.puncs:
            text = text.replace(punc, "\n")

        line_words = set()
        bline_words = set()
        for line in text.strip().split("\n"):
            line = line.strip()
            bline = line[::-1]

            line_gram = n_gram(line, max_size)
            for word in line_gram:
                # 主要时间消耗在 insert 上
                self.fw_ngram.insert(word)
                if len(word) > 1:
                    line_words.update({word})

            bline_gram = n_gram(bline, max_size)
            for word in bline_gram:
                self.bw_ngram.insert(word)
                if len(word) > 1:
                    bline_words.update({word})

        return line_words, bline_words

    def _process(self,
                 text,
                 max_size,
                 entropy_threshold,
                 mutualinfo_threshold,
                 freq_threshold,
                 top):
        fwords, bwords = self.preprocess(text, max_size)
        single_word = set(list(chain(*[list(word) for word in fwords])))

        # 互信息
        fw_mi = mutualinfo(fwords, self.fw_ngram)
        bw_mi = mutualinfo(bwords, self.bw_ngram)

        # 词左右信息熵
        fw_entropy = calc_entropy(fwords, self.fw_ngram)
        bw_entropy = calc_entropy(bwords, self.bw_ngram)

        # 字的左右信息熵
        fw_word_entropy = calc_entropy(single_word, self.fw_ngram)
        bw_word_entropy = calc_entropy(single_word, self.bw_ngram)

        # 取字的左右信息熵的最小值
        word_entropy = {key: min(fw_word_entropy[key], bw_word_entropy[key]) for key in fw_word_entropy}

        final = {}
        for k, v in fw_entropy.items():

            if k[::-1] in bw_mi and k in fw_mi:
                # 凝固度 最小值
                mi_min = min(fw_mi[k], bw_mi[k[::-1]])
                # 词频
                word_prob = max(self.fw_ngram.search(k)[1], self.bw_ngram.search(k[::-1])[1])
                if mi_min < mutualinfo_threshold:
                    continue
            else:
                continue
            # 增加 数字判断
            if word_prob < freq_threshold or k.isdigit():
                continue

            if k[::-1] in bw_entropy:
                en_min = min(v, bw_entropy[k[::-1]])
                if en_min < entropy_threshold:
                    continue
            else:
                continue
            # 字的左右信息熵之和
            sum_word_en = sum(word_entropy[key] for key in list(k))

            # （互信息 - 字信息熵 + 左右最小信息熵）* 词频
            score = (mi_min - sum_word_en + en_min) * word_prob
            # score = (mi_min, sum_word_en, en_min, word_prob)
            final[k] = score
        result = sorted(final.items(), key=lambda x: x[1], reverse=True)
        if top:
            result = result[: top]

        return result


class WordDiscovery(_WordDiscovery):

    # def __init__(self):
    #     pass

    def new_word(self,
                 text,
                 max_size=5,
                 entropy_threshold=0.04,
                 mutualinfo_threshold=4,
                 freq_threshold=50,
                 top=None,
                 history=0):
        """

        :param text: 要分词的文本，长文本字符串
        :param max_size: 最大切词长度
        :param entropy_threshold: 左右信息熵（自由度）阈值
        :param mutualinfo_threshold: 互信息（凝固度阈值）
        :param freq_threshold: 词频
        :param top: 选取top n 个词
        :param history: 是否基于存档数据对现有text进行切词，0 代表否， 1代表是。默认为0
        :return:
        """
        max_size = max_size + 1

        if history not in (0, 1):
            raise ValueError("history value must be 0 or 1")

        if history == 0:
            _worddis = _WordDiscovery()
            result = _worddis._process(text,
                                       max_size,
                                       entropy_threshold,
                                       mutualinfo_threshold,
                                       freq_threshold,
                                       top)
        else:
            with open('../data/fw_ngram.pkl', 'rb') as pf:
                self.fw_ngram = pickle.load(pf)
            with open('../data/bw_ngram.pkl', 'rb') as pf:
                self.bw_ngram = pickle.load(pf)
            result = super()._process(text,
                                      max_size,
                                      entropy_threshold,
                                      mutualinfo_threshold,
                                      freq_threshold,
                                      top
                                      )
        return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="the file to discover new word, must be txt or csv")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="discover new word to output")

    ## optional parameters
    parser.add_argument("--max_size",
                        default=None,
                        type=int,
                        help="max_size")

    parser.add_argument("--entropy_threshold",
                        default=None,
                        type=float,
                        help="entropy_threshold")

    parser.add_argument("--mutualinfo_threshold",
                        default=None,
                        type=int,
                        help="mutualinfo_threshold")

    parser.add_argument("--freq_threshold",
                        default=None,
                        type=int,
                        help="freq_threshold")

    parser.add_argument("--history",
                        default=None,
                        type=int,
                        help="history 0  without history, 1 with history ")

    args = parser.parse_args()

    # with open(args.input_file, encoding='gbk', errors='ignore') as f:
    #     texts = f.readlines()
    #
    # text_new = ' '.join(texts)
    data = pd.read_csv(open(args.input_file, encoding='utf-8'), nrows=10000)[['comment']]
    text = data['comment'].values
    text_new = ''.join(text)

    import time
    start = time.time()
    newWordDiscovery = WordDiscovery()
    results = newWordDiscovery.new_word(text_new,
                                        max_size=5,
                                        entropy_threshold=0.004,
                                        mutualinfo_threshold=2,
                                        freq_threshold=2,
                                        history=0)
    result =pd.DataFrame(results)
    result.to_csv(args.output_file, encoding='utf-8', index=False)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()
