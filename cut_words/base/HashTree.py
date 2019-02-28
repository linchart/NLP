# encoding: utf-8

"""

@author: linchart
@file: hash_tree.py
@version: 1.0
@time : 2018/11/27

"""


class Node:
    """
    初始化节点
    """
    def __init__(self):
        self.occupied = False
        self.tree = {}
        self.key = None


class HashTree:
    """
    根据质数分辨定理
    """
    def __init__(self):
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
                       47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
                       103, 107, 109, 113, 127, 131, 137, 139]
        self.node = Node()

    def insert(self, value):
        node = self.node
        count = 0
        while True:
            remain = value % self.primes[count]
            if remain not in node.tree:
                node.tree[remain] = Node()
                node = node.tree[remain]
                node.occupied = True
                node.key = value
                break
            else:
                node = node.tree[remain]
                count += 1

    def search(self, value):
        node = self.node
        count = 0
        while True:
            remain = value % self.primes[count]
            if remain in node.tree:
                node = node.tree[remain]
                if node.key != value:
                    count += 1
                    continue
                else:
                    return True
            else:
                return False