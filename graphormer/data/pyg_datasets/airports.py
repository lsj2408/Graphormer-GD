import os.path as osp
from typing import Callable, List, Optional

import torch
from torch_sparse import coalesce

from torch_geometric.data import Data, InMemoryDataset, download_url
import algos

import numpy as np
import networkx as nx

class Airports(InMemoryDataset):
    r"""The Airports dataset from the `"struc2vec: Learning Node
    Representations from Structural Identity"
    <https://arxiv.org/abs/1704.03165>`_ paper, where nodes denote airports
    and labels correspond to activity levels.
    Features are given by one-hot encoded node identifiers, as described in the
    `"GraLSP: Graph Neural Networks with Local Structural Patterns"
    ` <https://arxiv.org/abs/1911.07675>`_ paper.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"USA"`, :obj:`"Brazil"`,
            :obj:`"Europe"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    edge_url = ('https://github.com/leoribeiro/struc2vec/'
                'raw/master/graph/{}-airports.edgelist')
    label_url = ('https://github.com/leoribeiro/struc2vec/'
                 'raw/master/graph/labels-{}-airports.txt')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['usa', 'brazil', 'europe']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.__indices__ = range(self.data.y.shape[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [
            f'{self.name}-airports.edgelist',
            f'labels-{self.name}-airports.txt',
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.edge_url.format(self.name), self.raw_dir)
        download_url(self.label_url.format(self.name), self.raw_dir)

    def process(self):
        index_map, ys = {}, []
        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            for i, row in enumerate(data):
                idx, y = row.split()
                index_map[int(idx)] = i
                ys.append(int(y))
        y = torch.tensor(ys, dtype=torch.long)

        # remove one-hot node identifier
        # x = torch.eye(y.size(0))
        x = torch.ones(y.size(0), 1).long()

        edge_indices = []
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[:-1]
            for row in data:
                src, dst = row.split()
                edge_indices.append([index_map[int(src)], index_map[int(dst)]])
                edge_indices.append([index_map[int(dst)], index_map[int(src)]])
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_index, _ = coalesce(edge_index, None, y.size(0), y.size(0))

        # preprocessing
        N = x.size(0)

        # (1) node adj matrix [N, N] bool
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        # (2) resistance distance matrix
        g = nx.Graph(adj.float().numpy())
        g_components_list = [g.subgraph(c).copy() for c in nx.connected_components(g)]
        g_resistance_matrix = np.zeros((N, N)) - 1.0
        g_index = 0
        for item in g_components_list:
            cur_adj = nx.to_numpy_array(item)
            cur_num_nodes = cur_adj.shape[0]
            cur_res_dis = np.linalg.pinv(
                np.diag(cur_adj.sum(axis=-1)) - cur_adj + np.ones((cur_num_nodes, cur_num_nodes),
                                                                  dtype=np.float32) / cur_num_nodes
            ).astype(np.float32)
            A = np.diag(cur_res_dis)[:, None]
            B = np.diag(cur_res_dis)[None, :]
            cur_res_dis = A + B - 2 * cur_res_dis
            g_resistance_matrix[g_index:g_index + cur_num_nodes, g_index:g_index + cur_num_nodes] = cur_res_dis
            g_index += cur_num_nodes
        g_cur_index = []
        for item in g_components_list:
            g_cur_index.extend(list(item.nodes))
        g_resistance_matrix = g_resistance_matrix[g_cur_index, :]
        g_resistance_matrix = g_resistance_matrix[:, g_cur_index]

        if g_resistance_matrix.max() > N - 1:
            print(f'error: {g_resistance_matrix}')
        g_resistance_matrix[g_resistance_matrix == -1.0] = 512.0
        res_matrix = np.zeros((N, N), dtype=np.float32)
        res_matrix[:, :N] = g_resistance_matrix
        res_matrix = torch.from_numpy(res_matrix).float()

        # (3) shortest_path_distance matrix
        shortest_path_result, path = algos.floyd_warshall(adj.numpy())

        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        attn_bias = torch.zeros(
            [N + 1, N + 1], dtype=torch.float)  # with graph token
        in_degree = adj.long().sum(dim=1).view(-1)

        data = Data(x=x, edge_index=edge_index, y=y, attn_bias=attn_bias, spatial_pos=spatial_pos, in_degree=in_degree, out_degree=in_degree, res_pos=res_matrix)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Airports()'