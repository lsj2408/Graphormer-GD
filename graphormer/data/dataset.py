from functools import lru_cache

import ogb
import numpy as np
import torch
from torch.nn import functional as F
from fairseq.data import data_utils, FairseqDataset, BaseWrapperDataset

from .wrapper import MyPygGraphPropPredDataset
from .synthetic_wrapper import MyPygPCQM4MDatasetBridge, MyPygPCQM4MDatasetAP
from .collator import collator, collator_with_resistance_distance, collator_node, collator_ap, collator_bridge

from typing import Optional, Union
from torch_geometric.data import Data as PYGDataset
from dgl.data import DGLDataset
from .dgl_datasets import DGLDatasetLookupTable, GraphormerDGLDataset
from .pyg_datasets import PYGDatasetLookupTable, GraphormerPYGDataset
from .ogb_datasets import OGBDatasetLookupTable
from ogb.lsc import PCQM4Mv2Evaluator

class BatchedDataDataset(FairseqDataset):
    def __init__(
        self, dataset, max_node=128, multi_hop_max_dist=5, spatial_pos_max=1024, with_resistance_distance=False, with_airports=False, with_synthetic_ap=False, with_synthetic_bridge=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
        self.with_resistance_distance = with_resistance_distance
        self.with_airports = with_airports
        self.with_synthetic_ap = with_synthetic_ap
        self.with_synthetic_bridge = with_synthetic_bridge

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):

        if self.with_resistance_distance:
            return collator_with_resistance_distance(samples,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,)

        if self.with_airports:
            return collator_node(samples,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max)

        if self.with_synthetic_ap:
            return collator_ap(samples,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max)

        if self.with_synthetic_bridge:
            return collator_bridge(samples,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max)


        return collator(
            samples,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )


class TargetDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index].y

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.stack(samples, dim=0)


class GraphormerDataset:
    def __init__(
        self,
        dataset: Optional[Union[PYGDataset, DGLDataset]] = None,
        dataset_spec: Optional[str] = None,
        dataset_source: Optional[str] = None,
        seed: int = 0,
        train_idx = None,
        valid_idx = None,
        test_idx = None,
    ):
        super().__init__()
        if dataset is not None:
            if dataset_source == "dgl":
                self.dataset = GraphormerDGLDataset(dataset, seed=seed, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
            elif dataset_source == "pyg":
                self.dataset = GraphormerPYGDataset(dataset, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)
            else:
                raise ValueError("customized dataset can only have source pyg or dgl")
        elif dataset_source == "dgl":
            self.dataset = DGLDatasetLookupTable.GetDGLDataset(dataset_spec, seed=seed)
        elif dataset_source == "pyg":
            self.dataset = PYGDatasetLookupTable.GetPYGDataset(dataset_spec, seed=seed)
        elif dataset_source == "ogb":
            self.dataset = OGBDatasetLookupTable.GetOGBDataset(dataset_spec, seed=seed)
        self.setup()

    def setup(self):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data

class PygAirPreprocessedData():
    def __init__(self, dataset_name, dataset_path = "../dataset", seed=42):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset = PYGDatasetLookupTable.GetPYGDataset(dataset_name, seed=seed)
        self.max_node = self.dataset[0].x.shape[0] + 10
        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024
        self.loss_fn = F.l1_loss
        self.num_class = len(self.dataset[0].y.unique())
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator()
        self.setup()

    def setup(self, stage: str = None):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data


class SyntheticPreprocessedData():
    def __init__(self, dataset_name, dataset_path = "../dataset", seed=42):
        super().__init__()

        assert dataset_name in [
            "Synthetic-Anticulation-Point",
            "Synthetic-Bridge"
        ], "Only support Anticulation-Point & Bridge detection"
        self.dataset_name = dataset_name
        if dataset_name == 'Synthetic-Anticulation-Point':
            self.dataset = MyPygPCQM4MDatasetAP(root=dataset_path)
        else:
            self.dataset = MyPygPCQM4MDatasetBridge(root=dataset_path)
        self.setup()

    def setup(self, stage: str = None):
        split_idx = self.dataset.get_idx_split()
        self.train_idx = split_idx["train"]
        self.valid_idx = split_idx["valid"]
        self.test_idx = split_idx["test-dev"]

        self.dataset_train = self.dataset.index_select(self.train_idx)
        self.dataset_val = self.dataset.index_select(self.valid_idx)
        self.dataset_test = self.dataset.index_select(self.test_idx)

        self.max_node = 128
        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024
        self.loss_fn = F.l1_loss
        self.num_class = 1
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator(),


class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, num_samples, seed):
        super().__init__(dataset)
        self.num_samples = num_samples
        self.seed = seed
        self.set_epoch(1)

    def set_epoch(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.random.permutation(self.num_samples)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False
