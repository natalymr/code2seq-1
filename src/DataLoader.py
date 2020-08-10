from typing import List

import numpy as np

import os
import time
import torch
import random

from sklearn.utils import shuffle

from .utils import pad_seq, sentence_to_ids


class DataLoader(object):

    def __init__(self, data_path, batch_size, num_k, vocab_subtoken, vocab_nodes, vocab_target, device, shuffle=True,
                 batch_time=False):

        """
        data_path : path for data
        num_examples : total lines of data file
        batch_size : batch size
        num_k : max ast paths included to one examples
        vocab_subtoken : dict of subtoken and its id (leaves)
        vocab_nodes : dict of node symbol and its id (nodes)
        vocab_target : dict of target symbol and its id (function names subtokens)
        """

        self.data_path = data_path
        self.batch_size = batch_size

        self.num_examples = self.file_count(data_path)
        self.num_k = num_k

        self.vocab_subtoken = vocab_subtoken
        self.vocab_nodes = vocab_nodes
        self.vocab_target = vocab_target

        self.index = 0
        self.pointer = np.array(range(self.num_examples))
        self.shuffle = shuffle

        self.batch_time = batch_time

        self.device = device

        self.reset()

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch_time:
            t1 = time.time()

        if self.index >= self.num_examples:
            self.reset()
            raise StopIteration()

        ids = self.pointer[self.index: self.index + self.batch_size]
        seqs_s, seqs_n, seqs_e, seqs_y = self.read_batch(ids)

        # length_k : (batch_size, k) для каждого в батче смотрим, сколько было путей для каждой функции
        lengths_k = [len(ex) for ex in seqs_n]

        # flattening (batch_size, k, l) to (batch_size * k, l)
        # this is useful to make torch.tensor
        # лист листов, где каждый лист - это номера сабтокенов у листов
        seqs_s = [symbol for k in seqs_s for symbol in k]
        # лист листов, где каждый лист - это номера вершин в путях
        seqs_n = [symbol for k in seqs_n for symbol in k]
        # лист листов, где каждый лист - это номера вершин в путях
        seqs_e = [symbol for k in seqs_e for symbol in k]

        # Padding
        lengths_s = [len(s) for s in seqs_s]  # количество сабтокенов в каждом листе
        lengths_n = [len(s) for s in seqs_n]  # количество вершин в каждом пути
        lengths_e = [len(s) for s in seqs_e]  # количество сабтокенов в каждом листе
        lengths_y = [len(s) for s in seqs_y]  # количество сабтокенов в каждой таргете, т.е. функции

        max_length_s = max(lengths_s)
        max_length_n = max(lengths_n)
        max_length_e = max(lengths_e)
        max_length_y = max(lengths_y)

        # лист листов: (batch_size * path_number * List[номера сабтокенов у листов_слева, PAD])
        padded_s = [pad_seq(s, max_length_s) for s in seqs_s]
        # лист листов: (batch_size * path_number * List[номера вершин в путях, PAD])
        padded_n = [pad_seq(s, max_length_n) for s in seqs_n]

        # лист листов: (batch_size * path_number * List[номера сабтокенов у листов_справа, PAD])
        padded_e = [pad_seq(s, max_length_e) for s in seqs_e]
        # лист листов: (batch_size * list[номера сабтокенов у таргета, PAD])
        padded_y = [pad_seq(s, max_length_y) for s in seqs_y]

        # index for split (batch_size * k, l) into (batch_size, k, l)
        index_N = range(len(lengths_n))  # range(0, количество_путей_в_батче)

        # sort for rnn
        seq_pairs = sorted(zip(lengths_n, index_N, padded_n, padded_s, padded_e), key=lambda p: p[0], reverse=True)
        lengths_n, index_N, padded_n, padded_s, padded_e = zip(*seq_pairs)

        batch_S = torch.tensor(padded_s, dtype=torch.long, device=self.device)
        batch_E = torch.tensor(padded_e, dtype=torch.long, device=self.device)
        # transpose for rnn
        batch_N = torch.tensor(padded_n, dtype=torch.long, device=self.device).transpose(0, 1)
        batch_Y = torch.tensor(padded_y, dtype=torch.long, device=self.device).transpose(0, 1)
        """
        Example for BS = 2:
            Было: padded_y = [[6, 165, 2], [108, 172, 2]] 
            Стало: batch_Y = 
            tensor([[  6, 108],
                    [165, 172],
                    [  2,   2]])
            Было: padded_n = ([17, 83, 105, 11, 44, 7, 2], [56, 11, 5, 4, 10, 17, 2], [7, 105, 11, 44, 7, 2, 0],
            Стало: batch_N = 
            tensor([[ 17,  56,   7,   7,   7,   7,   7,  56],
                    [ 83,  11, 105, 105, 105,  44,  44,  11],
                    [105,   5,  11,  83,  11,  11,  11,   9],
                    [ 11,   4,  44,  17,  47,  56,   9,   2],
                    [ 44,  10,   7,   2,   2,   2,   2,   0],
        """


        # update index
        self.index += self.batch_size

        if self.batch_time:
            t2 = time.time()
            elapsed_time = t2 - t1
            print(f"batching time：{elapsed_time}")

        return batch_S, batch_N, batch_E, batch_Y, \
               lengths_s, lengths_n, lengths_e, lengths_y, \
               max_length_s, max_length_n, max_length_e, max_length_y, \
               lengths_k, index_N

    def reset(self):
        if self.shuffle:
            self.pointer = shuffle(self.pointer)
        self.index = 0

    def file_count(self, path):
        lst = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
        return len(lst)

    def read_batch(self, ids):
        seqs_s: List[List[List[int]]] = []
        seqs_e: List[List[List[int]]] = []
        seqs_n: List[List[List[int]]] = []
        seqs_y: List[List[int]] = []

        for i in ids:
            path = self.data_path + '/{:0>6d}.txt'.format(i)
            # print(path)
            with open(path, 'r') as f:
                seq_s: List[List[int]] = []
                seq_n: List[List[int]] = []
                seq_e: List[List[int]] = []

                target, *syntax_path = f.readline().split(' ')
                target = target.split('|')
                target = sentence_to_ids(self.vocab_target, target)

                # remove '' and '\n' in sequence, java-small dataset contains many '' in a line.
                syntax_path = [s for s in syntax_path if s != '' and s != '\n']

                # if the amount of ast path exceed the k,
                # uniformly sample ast pathes, as described in the paper.
                if len(syntax_path) > self.num_k:
                    sampled_path_index = random.sample(range(len(syntax_path)), self.num_k)
                else:
                    sampled_path_index = range(len(syntax_path))

                for j in sampled_path_index:
                    terminal1, ast_path, terminal2 = syntax_path[j].split(',')
                    terminal1 = sentence_to_ids(self.vocab_subtoken, terminal1.split('|'))
                    ast_path = sentence_to_ids(self.vocab_nodes, ast_path.split('|'))
                    terminal2 = sentence_to_ids(self.vocab_subtoken, terminal2.split('|'))

                    seq_s.append(terminal1)
                    seq_e.append(terminal2)
                    seq_n.append(ast_path)

                seqs_s.append(seq_s)
                seqs_e.append(seq_e)
                seqs_n.append(seq_n)
                seqs_y.append(target)
        return seqs_s, seqs_n, seqs_e, seqs_y
