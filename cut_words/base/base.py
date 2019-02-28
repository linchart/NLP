# encoding: utf-8

"""

@author: linchart
@file: base.py
@version: 1.0
@time : 2018/11/28

"""

import hashlib

from HashTree import HashTree


def get_key_words(filename):
    with open(filename, encoding='utf-8') as f:
        key_words = [word.replace('\n', '') for word in f.readlines()]
        key_words_hash = [int(hashlib.sha1(word.encode('utf-8')).hexdigest(), 16)
                          for word in key_words]
    max_len = max([len(word) for word in key_words])
    return key_words_hash, max_len


def load_hash(hash_words):
    hash_tree_ = HashTree()
    for word in hash_words:
        hash_tree_.insert(word)
    return hash_tree_


class WordsSplit:
    """
    包括最大正向匹配、最大逆向匹配、双向匹配
    """

    def __init__(self, file):
        self.keywords, self.max_len = get_key_words(file)
        self.hash_tree_ = load_hash(self.keywords)

    # def load_tree(self, hash_key):
    #     for word in hash_key:
    #         self.hash_tree_.insert(word)
    #     print("hash tree 加载完成")

    def _words_match(self, string, method):
        """
        :param string: 待切分字符串
        :param method: 匹配方法，FMM 为最大正向匹配，BMM 为最大逆向匹配，BM 为双向匹配
        :return:
        """
        string = string.strip()
        string_len = len(string)
        left_string_len = string_len
        splited_string = ""
        splited_len = 0
        result = []
        while left_string_len > 0:
            split_len = min(self.max_len, string_len)
            if method == 'FMM':
                string_split = string[splited_len: splited_len + split_len]
            elif method == 'BMM':
                end = string_len - splited_len
                string_split = string[max(0, end - split_len): end]

            hash_string = int(hashlib.sha1(string_split.encode('utf-8')).hexdigest(), 16)

            while not self.hash_tree_.search(hash_string):
                if len(string_split) == 1:
                    break
                else:
                    if method == 'FMM':
                        string_split = string_split[:len(string_split) - 1]
                    elif method == 'BMM':
                        string_split = string_split[1:]
                    hash_string = int(hashlib.sha1(string_split.encode('utf-8')).hexdigest(), 16)
            result.append(string_split)
            splited_string = splited_string + string_split
            splited_len = len(splited_string)
            left_string_len = string_len - splited_len
        else:
            if method == 'FMM':
                return result
            elif method == 'BMM':
                return result[::-1]

    def search_word(self, word_set):
        count = 0
        for word in word_set:
            if len(word) == 1:
                word_string = int(hashlib.sha1(word.encode('utf-8')).hexdigest(), 16)
                if self.hash_tree_.search(word_string):
                    count += 1
        return count


    def words_match(self, string, method="FMM"):

        if method not in ("FMM", "BMM", "BM"):
            raise Exception("method must be one of FMM, BMM, BM ")

        if method == "FMM":
            return ' '.join(self._words_match(string, method="FMM"))
        elif method == "BMM":
            return ' '.join(self._words_match(string, method="BMM"))
        else:
            fmm = self._words_match(string, method="FMM")
            bmm = self._words_match(string, method="BMM")
            fmm_len = len(fmm)
            bmm_len = len(bmm)

            if fmm_len < bmm_len:
                return ' '.join(fmm)
            elif fmm_len > bmm_len:
                return ' '.join(bmm)
            elif fmm == bmm:
                return ' '.join(bmm)
            else:
                # 如果正向 和 逆向 切分词数相等，则切分的词中，单个词在词典中的数量越多，效果越好, 如果在词典中
                # 词的数量一致，则随机选一个
                # single_fmm = len([word for word in fmm if len(word) == 1 and word in self.keywords])
                # single_bmm = len([word for word in bmm if len(word) == 1 and word in self.keywords])
                single_fmm = self.search_word(fmm)
                single_bmm = self.search_word(bmm)
                single_result = {single_fmm: fmm, single_bmm: bmm}
                result = single_result.get(min(single_fmm, single_bmm))
                return ' '.join(result)


if __name__ == '__main__':
    file_ = '../input/new_dict.txt'
    cut_words = WordsSplit(file_)
    test_file = r"E:\work\NLP研究应用\分词\input\icwb2-data\testing\msr_test.txt"
    data = []
    with open(test_file, encoding='gbk') as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.replace("\n", ""))
    import time
    start = time.time()
    result = []
    for test in data:
        result.append(cut_words.words_match(test, method="BM"))
    end = time.time()
    print(end - start)
    valid_file = r"E:\work\NLP研究应用\分词\input\icwb2-data\gold\msr_test_gold.txt"
    result_valid = []
    with open(valid_file, encoding='gbk') as f:
        lines = f.readlines()
        for line in lines:
            result_valid.append(line.replace("\n", ""))
