from dataclasses import dataclass
from typing import List

import numpy as np

import os
import time
import torch
import random

from sklearn.utils import shuffle

from .utils import pad_seq, pad_seq_new_tokenizer, sentence_to_ids

# @dataclass
# class BatchTwoInput:
#     return {
#         'del_left_leaves': tensor_del_left_l,
#         'del_nodes': tensor_del_nodes,
#         'del_right_leaves': tensor_del_right_l,
#         'add_left_leaves': tensor_add_left_l,
#         'add_nodes': tensor_add_nodes,
#         'add_right_leaves': tensor_add_right_l,
#         'targets': tensor_targets,
#
#         'lens_del_left_leaves': lens_b_del_left_l,
#         'lens_del_nodes': lens_b_del_nodes,
#         'lens_del_right_leaves': lens_b_del_right_l,
#         'lens_add_left_leaves': lens_b_add_left_l,
#         'lens_add_nodes': lens_b_add_nodes,
#         'lens_add_right_leaves': lens_b_add_right_l,
#         'lens_targets': lengths_b_targets,
#
#         'permutation_index_del': perm_index_del,
#         'permutation_index_add': perm_index_add,
#         'len_k_del': lengths_del_k,
#         'len_k_add': lengths_add_k
#     }


class DataLoaderTwoInput(object):

    def __init__(self, data_path, batch_size, max_paths_number,
                 vocab_subtoken, vocab_nodes, vocab_target,
                 device, shuffle=True, batch_time=False):

        """
        data_path : path for data
        num_examples : total lines of data file
        batch_size : batch size
        max_paths_number : max ast paths included to one examples
        vocab_subtoken : dict of subtoken and its id (leaves)
        vocab_nodes : dict of node symbol and its id (nodes)
        vocab_target : dict of target symbol and its id (function names subtokens)
        """

        self.data_path = data_path
        self.batch_size = batch_size

        self.num_examples = self.file_count(data_path)
        self.max_paths_number = max_paths_number

        self.vocab_subtoken = vocab_subtoken
        self.vocab_nodes = vocab_nodes
        self.vocab_target = vocab_target
        self.eos_token = self.vocab_target(self.vocab_target.eos_token)['input_ids'][0]

        self.index = 0
        self.pointer = np.array(range(self.num_examples))
        self.shuffle = shuffle

        self.batch_time = batch_time
        self.device = device

        self.reset()

    def __iter__(self):
        return self

    @staticmethod
    def padding_data_pipeline(data: List[List[List[int]]]) -> (List[List[int]], List[int], int):
        # flattening (batch_size, k, l) to (batch_size * k, l), this is useful to make torch.tensor
        data = [symbol for path in data for symbol in path]
        lengths = [len(sub_list) for sub_list in data]
        max_length = max(lengths)
        padded_data = [pad_seq(sub_list, max_length) for sub_list in data]
        return padded_data, lengths, max_length

    @staticmethod
    def padding_target_pipeline(data: List[List[int]]) -> (List[List[int]], List[int], int):
        # flattening (batch_size, k, l) to (batch_size * k, l), this is useful to make torch.tensor
        lengths = [len(sub_list) for sub_list in data]
        max_length = max(lengths)
        padded_data = [pad_seq_new_tokenizer(sub_list, max_length) for sub_list in data]
        return padded_data, lengths, max_length

    def __next__(self):

        if self.batch_time:
            t1 = time.time()

        if self.index >= self.num_examples:
            self.reset()
            raise StopIteration()

        ids = self.pointer[self.index: self.index + self.batch_size]
        b_del_left_leaves, b_del_nodes, b_del_right_leaves, \
            b_add_left_leaves, b_add_nodes, b_add_right_leaves, b_targets, file_number = self.read_batch(ids)

        # length_k : (batch_size, k) для каждого в батче смотрим, сколько было путей для каждой функции
        lengths_del_k = [len(cur_line_paths) for cur_line_paths in b_del_nodes]
        lengths_add_k = [len(cur_line_paths) for cur_line_paths in b_add_nodes]

        # Padding
        # padded_b_l_leaves - лист листов: (batch_size * path_number * List[номера сабтокенов у листов_слева, PAD])
        # lengths_b_l_leaves - количество сабтокенов в каждом листе
        # padded_b_nodes - лист листов: (batch_size * path_number * List[номера вершин в путях, PAD])
        # lengths_b_nodes - количество вершин в каждом пути
        padded_b_targets, lengths_b_targets, max_length_b_targets = self.padding_target_pipeline(b_targets)

        padded_b_del_left_l, lens_b_del_left_l, max_len_b_del_left_l = self.padding_data_pipeline(b_del_left_leaves)
        padded_b_del_right_l, lens_b_del_right_l, max_len_b_del_right_l = self.padding_data_pipeline(b_del_right_leaves)
        padded_b_del_nodes, lens_b_del_nodes, max_len_b_del_nodes = self.padding_data_pipeline(b_del_nodes)

        padded_b_add_left_l, lens_b_add_left_l, max_len_b_add_left_l = self.padding_data_pipeline(b_add_left_leaves)
        padded_b_add_right_l, lens_b_add_right_l, max_len_b_add_right_l = self.padding_data_pipeline(b_add_right_leaves)
        padded_b_add_nodes, lens_b_add_nodes, max_len_b_add_nodes = self.padding_data_pipeline(b_add_nodes)

        # index for split (batch_size * k, l) into (batch_size, k, l)
        perm_index_del = range(len(lens_b_del_nodes))  # range(0, количество_путей_в_батче)
        perm_index_add = range(len(lens_b_add_nodes))  # range(0, количество_путей_в_батче)

        # sort for rnn: Del
        lens_b_del_nodes, perm_index_del, padded_b_del_nodes, \
            padded_b_del_left_l, padded_b_del_right_l = self.sort_by_len(lens_b_del_nodes, perm_index_del,
                                                                         padded_b_del_nodes, padded_b_del_left_l,
                                                                         padded_b_del_right_l)
        # sort for rnn: Add
        lens_b_add_nodes, perm_index_add, padded_b_add_nodes, \
            padded_b_add_left_l, padded_b_add_right_l = self.sort_by_len(lens_b_add_nodes, perm_index_add,
                                                                         padded_b_add_nodes, padded_b_add_right_l,
                                                                         padded_b_add_left_l)

        tensor_del_left_l = torch.tensor(padded_b_del_left_l, dtype=torch.long, device=self.device)
        tensor_del_right_l = torch.tensor(padded_b_del_right_l, dtype=torch.long, device=self.device)

        tensor_add_left_l = torch.tensor(padded_b_add_left_l, dtype=torch.long, device=self.device)
        tensor_add_right_l = torch.tensor(padded_b_add_right_l, dtype=torch.long, device=self.device)

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
        # transpose for rnn
        tensor_del_nodes = torch.tensor(padded_b_del_nodes, dtype=torch.long, device=self.device).transpose(0, 1)
        tensor_add_nodes = torch.tensor(padded_b_add_nodes, dtype=torch.long, device=self.device).transpose(0, 1)
        tensor_targets = torch.tensor(padded_b_targets, dtype=torch.long, device=self.device).transpose(0, 1)

        # update index
        self.index += self.batch_size

        if self.batch_time:
            t2 = time.time()
            elapsed_time = t2 - t1
            print(f"batching time：{elapsed_time}")

        # return tensor_del_left_l, tensor_del_nodes, tensor_del_right_l, tensor_targets, \
        #        lens_b_del_left_l, lens_b_del_nodes, lens_b_del_right_l, lengths_b_targets, \
        #        max_len_b_del_left_l, max_len_b_del_nodes, max_len_b_del_right_l, max_length_b_targets, \
        #        lengths_del_k, perm_index_del
        return {
            'del_left_leaves': tensor_del_left_l,
            'del_nodes': tensor_del_nodes,
            'del_right_leaves': tensor_del_right_l,
            'add_left_leaves': tensor_add_left_l,
            'add_nodes': tensor_add_nodes,
            'add_right_leaves': tensor_add_right_l,
            'targets': tensor_targets,

            'lens_del_left_leaves': lens_b_del_left_l,
            'lens_del_nodes': lens_b_del_nodes,
            'lens_del_right_leaves': lens_b_del_right_l,
            'lens_add_left_leaves': lens_b_add_left_l,
            'lens_add_nodes': lens_b_add_nodes,
            'lens_add_right_leaves': lens_b_add_right_l,
            'lens_targets': lengths_b_targets,

            'permutation_index_del': perm_index_del,
            'permutation_index_add': perm_index_add,
            'len_k_del': lengths_del_k,
            'len_k_add': lengths_add_k,

            'file_number': file_number
        }

    def sort_by_len(self, lens_of_paths, permutation_index, nodes, right_leaves, left_leaves):
        seq_pairs = sorted(zip(lens_of_paths, permutation_index, nodes, left_leaves, right_leaves),
                           key=lambda p: p[0],
                           reverse=True)
        return zip(*seq_pairs)

    def reset(self):
        if self.shuffle:
            self.pointer = shuffle(self.pointer)
        self.index = 0

    def file_count(self, path):
        lst = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
        return len(lst)

    def read_batch(self, batch_ids):
        batch_del_left_leaves, batch_del_nodes, batch_del_right_leaves = [], [], []  # List[List[List[int]]]
        batch_add_left_leaves, batch_add_nodes, batch_add_right_leaves = [], [], []  # List[List[List[int]]]
        batch_targets: List[List[int]] = []
        file_number = ""

        for cur_id in batch_ids:
            path = self.data_path + '/{:0>6d}.txt'.format(cur_id)
            file_number = '/{:0>6d}.txt'.format(cur_id)
            with open(path, 'r') as f:
                target, del_paths, add_paths = f.readline().split('\t')
                target = target.split('|')
                target = " ".join(target)
#                print(target)
                # target, commit = [target[0]], target[0]
                target = self.vocab_target(target)['input_ids']
                target.append(self.eos_token)
#                print(target)                
#                print(self.vocab_target.convert_tokens_to_string(self.vocab_target.convert_ids_to_tokens(target)))#.encode('utf-8'))#.decode('ascii'))
#sentence_to_ids(self.vocab_target, target)

                del_paths, add_paths = del_paths.split(), add_paths.split()

                del_left_leaves, del_nodes, del_right_leaves = self.extract_paths_from_line(del_paths)
                add_left_leaves, add_nodes, add_right_leaves = self.extract_paths_from_line(add_paths)

                batch_del_left_leaves.append(del_left_leaves)
                batch_del_nodes.append(del_nodes)
                batch_del_right_leaves.append(del_right_leaves)

                batch_add_left_leaves.append(add_left_leaves)
                batch_add_nodes.append(add_nodes)
                batch_add_right_leaves.append(add_right_leaves)

                batch_targets.append(target)
        return batch_del_left_leaves, batch_del_nodes, batch_del_right_leaves, \
            batch_add_left_leaves, batch_add_nodes, batch_add_right_leaves, \
            batch_targets, file_number

    def extract_paths_from_line(self, paths: List[str]) -> (List[int], List[int], List[int]):
        # remove '' and '\n' in sequence, java-small dataset contains many '' in a line.
        paths = [s for s in paths if s != '' and s != '\n']
        left_leaves: List[List[int]] = []
        right_leaves: List[List[int]] = []
        nodes: List[List[int]] = []

        # if the amount of ast path exceed the k,
        # uniformly sample ast pathes, as described in the paper.
        if len(paths) > self.max_paths_number:
            sampled_path_index = random.sample(range(len(paths)), self.max_paths_number)
        else:
            sampled_path_index = range(len(paths))
        for j in sampled_path_index:
            terminal_left, ast_path, terminal_right = paths[j].split(',')
            terminal_left = sentence_to_ids(self.vocab_subtoken, terminal_left.split('|'))
            ast_path = sentence_to_ids(self.vocab_nodes, ast_path.split('|'))
            terminal_right = sentence_to_ids(self.vocab_subtoken, terminal_right.split('|'))

            left_leaves.append(terminal_left)
            nodes.append(ast_path)
            right_leaves.append(terminal_right)
        return left_leaves, nodes, right_leaves
