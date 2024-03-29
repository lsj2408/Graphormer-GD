# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split
from typing import List
import torch
import numpy as np

from ..wrapper import preprocess_item, preprocess_item_with_resistance_distance
from .. import algos

import copy
from functools import lru_cache


class GraphormerPYGDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        seed: int = 0,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
        train_set=None,
        valid_set=None,
        test_set=None,
        with_resistance_distance=False,
        with_airports=False,
    ):
        self.dataset = dataset
        self.with_resistance_distance = with_resistance_distance
        self.with_airports = with_airports
        if self.dataset is not None:
            if with_airports:
                self.num_data = self.dataset.data.y.shape[0]
            else:
                self.num_data = len(self.dataset)
        self.seed = seed

        if with_resistance_distance:
            self.preprocess_func = preprocess_item_with_resistance_distance
        else:
            self.preprocess_func = preprocess_item

        if train_idx is None and train_set is None:
            train_valid_idx, test_idx = train_test_split(
                np.arange(self.num_data),
                test_size=self.num_data // 10,
                random_state=seed,
            )
            if with_airports:
                train_idx, valid_idx = train_test_split(
                    train_valid_idx, test_size=self.num_data // 10, random_state=seed
                )
            else:
                train_idx, valid_idx = train_test_split(
                    train_valid_idx, test_size=self.num_data // 5, random_state=seed
                )
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        elif train_set is not None:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
            self.train_idx = None
            self.valid_idx = None
            self.test_idx = None
        else:
            self.num_data = len(train_idx) + len(valid_idx) + len(test_idx)
            self.train_idx = train_idx
            self.valid_idx = valid_idx
            self.test_idx = test_idx
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        self.__indices__ = None

    def index_select(self, idx):
        dataset = copy.copy(self)
        dataset.dataset = self.dataset.index_select(idx)
        if isinstance(idx, torch.Tensor):
            dataset.num_data = idx.size(0)
        else:
            dataset.num_data = idx.shape[0]
        dataset.__indices__ = idx
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    def create_subset(self, subset):
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.__indices__ = None
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            if self.with_airports:
                item = self.dataset.get(0)
                item.idx = self.indices()[idx]
                item.y = item.y.reshape(-1)
            else:
                item = self.dataset[idx]
                item.idx = idx
                item.y = item.y.reshape(-1)
                item = self.preprocess_func(item)
            return item
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return self.num_data
