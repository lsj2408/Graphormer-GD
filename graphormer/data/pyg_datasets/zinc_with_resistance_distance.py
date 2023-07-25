import os
import os.path as osp
import shutil
import pickle

import numpy as np
import networkx as nx
import torch
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)


class ZINC_RD(InMemoryDataset):
    r"""The ZINC dataset from the `"Grammar Variational Autoencoder"
    <https://arxiv.org/abs/1703.01925>`_ paper, containing about 250,000
    molecular graphs with up to 38 heavy atoms.
    The task is to regress a molecular property known as the constrained
    solubility.

    Args:
        root (string): Root directory where the dataset should be saved.
        subset (boolean, optional): If set to :obj:`True`, will only load a
            subset of the dataset (12,000 molecular graphs), following the
            `"Benchmarking Graph Neural Networks"
            <https://arxiv.org/abs/2003.00982>`_ paper. (default: :obj:`False`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')

    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        assert split in ['train', 'val', 'test']
        super(ZINC_RD, self).__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
            'val.index', 'test.index'
        ]

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
        os.unlink(path)

        for split in ['train', 'val', 'test']:
            download_url(self.split_url.format(split), self.raw_dir)

    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)

            indices = range(len(mols))

            if self.subset:
                with open(osp.join(self.raw_dir, f'{split}.index'), 'r') as f:
                    indices = [int(x) for x in f.read()[:-1].split(',')]

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                mol = mols[idx]

                x = mol['atom_type'].to(torch.long).view(-1, 1)
                y = mol['logP_SA_cycle_normalized'].to(torch.float)

                adj = mol['bond_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)

                # @ Roger added: resistance distance
                # 1) adjacency matrix
                N = x.shape[0]
                adj = np.zeros((N, N), dtype=np.float32)
                adj[edge_index[0, :], edge_index[1, :]] = 1.0

                # 2) connected_components
                g = nx.Graph(adj)
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
                ori_idx = np.arange(N)
                g_resistance_matrix[g_cur_index, :] = g_resistance_matrix[ori_idx, :]
                g_resistance_matrix[:, g_cur_index] = g_resistance_matrix[:, ori_idx]

                if g_resistance_matrix.max() > N - 1:
                    print(f'error: {g_resistance_matrix}')
                g_resistance_matrix[g_resistance_matrix == -1.0] = 512.0
                res_matrix = np.zeros((N, 51), dtype=np.float32)
                res_matrix[:, :N] = g_resistance_matrix


                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y, res_pos=torch.from_numpy(res_matrix))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
