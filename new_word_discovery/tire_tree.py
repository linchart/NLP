# encoding: utf-8

"""

@author: linchart
@file: tire_tree.py
@version: 1.0
@time : 2018/11/19

"""


class TireNode:
    """参数初始化 is_word 判断该节点是否是某个词的词尾 count 词频统计"""
    def __init__(self):
        self.tree = {}
        self.is_word = False
        self.count = 0

    def fetch(self, char):
        return self.tree[char]


class Tire:

    def __init__(self):
        self.root = TireNode()

    # 插入字串
    def insert(self, chars):
        node = self.root
        node.count += 1

        for char in chars:
            # node = node.tree.get(char, TireNode())
            if char not in node.tree:
                node.tree[char] = TireNode()
            node = node.tree[char]

        node.is_word = True
        node.count += 1

    def search(self, chars):
        node = self.root

        for char in chars:
            node = node.tree.get(char)
            if not node:
                return False
        return node.is_word, node.count

    def fetch(self, word):
        node = self.root
        for char in word:
            node = node.fetch(char)
        return node
