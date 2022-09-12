
import numpy as np
import torch
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

from typing import Optional, Callable, List
import os.path as osp

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

import random

class PHVDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_str = self.data[index]
        item_list = item_str.split('\t')
        seq1 = [int(i) for i in item_list[0].split(',')]
        len1 = len(seq1)
        seq2 = [int(i) for i in item_list[1].split(',')]
        len2 = len(seq2)
        vec1 = [float(i) for i in item_list[2].split(',')]
        vec2 = [float(i) for i in item_list[3].split(',')]
        relation = int(item_list[4])
        sample = {
            'seq1': torch.from_numpy(np.array(seq1)),
            'seq2': torch.from_numpy(np.array(seq2)),
            'vec1': torch.from_numpy(np.array(vec1)),
            'vec2': torch.from_numpy(np.array(vec2)),
            'len1': torch.tensor(len1),
            'len2': torch.tensor(len2),
            'relation': torch.tensor(relation)
        }
        return sample

def collate_func(batch_dict):
    seq1_batch = []
    seq2_batch = []
    vec1_batch = []
    vec2_batch = []
    len1_batch = []
    len2_batch = []
    relation_batch = []
    for i in range(len(batch_dict)):
        item = batch_dict[i]
        seq1_batch.append(item['seq1'])
        seq2_batch.append(item['seq2'])
        vec1_batch.append(item['vec1'])
        vec2_batch.append(item['vec2'])
        len1_batch.append(item['len1'])
        len2_batch.append(item['len2'])
        relation_batch.append(item['relation'])

    seq1_batch = pad_sequence(seq1_batch, batch_first=True)
    seq2_batch = pad_sequence(seq2_batch, batch_first=True)
    res = {}
    res['seq1'] = seq1_batch
    res['seq2'] = seq2_batch
    res['vec1'] = vec1_batch
    res['vec2'] = vec2_batch
    res['len1'] = len1_batch
    res['len2'] = len2_batch
    res['relation'] = relation_batch
    return res

def read_graph(folder):
    node_file = osp.join(folder, 'all_seq.txt')
    edge_file = osp.join(folder, 'all_edge.txt')
    train_edge_file = osp.join(folder, 'train_edge.txt')
    test_edge_file = osp.join(folder, 'test_edge.txt')
    node_list = []
    feature_len = []
    with open(node_file, 'r') as f:
        for line in f:
            line = line.strip()
            features = line.split(' ')
            features = [int(i) for i in features]
            feature_len.append(len(features))
            node_list.append(torch.from_numpy(np.array(features)))

    node_feature = pad_sequence(node_list, batch_first=True)
    seq_len = torch.from_numpy(np.array(feature_len))

    edge_list = []
    with open(edge_file, 'r') as f:
        for line in f:
            line = line.strip()
            node_ids = line.split(' ')
            node_ids = [int(i) for i in node_ids]
            edge_list.append(node_ids)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    train_edge_list = []
    with open(train_edge_file, 'r') as f:
        for line in f:
            line = line.strip()
            node_ids = line.split(' ')
            node_ids = [int(i) for i in node_ids]
            train_edge_list.append(node_ids)
    
    train_edge_index = torch.tensor(train_edge_list, dtype=torch.long).t().contiguous()

    test_edge_list = []
    with open(test_edge_file, 'r') as f:
        for line in f:
            line = line.strip()
            node_ids = line.split(' ')
            node_ids = [int(i) for i in node_ids]
            test_edge_list.append(node_ids)

    test_edge_index = torch.tensor(test_edge_list, dtype=torch.long).t().contiguous()

    data = Data(x=node_feature, edge_index=edge_index, train_edge_index=train_edge_index, test_edge_index=test_edge_index, seq_len=seq_len)
    return data

def get_graph(folder):
    node_file = osp.join(folder, 'all_seq.txt')
    edge_file = osp.join(folder, 'all_edge.txt')
    train_edge_file = osp.join(folder, 'train_edge.txt')
    test_edge_file = osp.join(folder, 'test_edge.txt')
    node_list = []
    feature_len = []
    with open(node_file, 'r') as f:
        for line in f:
            line = line.strip()
            features = line.split(' ')
            features = [int(i) for i in features]
            feature_len.append(len(features))
            node_list.append(torch.from_numpy(np.array(features)))

    node_feature = pad_sequence(node_list, batch_first=True)
    seq_len = torch.from_numpy(np.array(feature_len))

    edge_list = []
    with open(edge_file, 'r') as f:
        for line in f:
            line = line.strip()
            node_ids = line.split(' ')
            node_ids = [int(i) for i in node_ids]
            edge_list.append(node_ids)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    train_edge_list = []
    with open(train_edge_file, 'r') as f:
        for line in f:
            line = line.strip()
            node_ids = line.split(' ')
            node_ids = [int(i) for i in node_ids]
            train_edge_list.append(node_ids)
    
    train_edge_index = torch.tensor(train_edge_list, dtype=torch.long).t().contiguous()

    test_edge_list = []
    with open(test_edge_file, 'r') as f:
        for line in f:
            line = line.strip()
            node_ids = line.split(' ')
            node_ids = [int(i) for i in node_ids]
            test_edge_list.append(node_ids)

    test_edge_index = torch.tensor(test_edge_list, dtype=torch.long).t().contiguous()

    all_data = Data(x=node_feature, edge_index=edge_index, seq_len=seq_len)
    return all_data


class GraphDataset(InMemoryDataset):
    url = ""

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):

        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        names = ["dgraphfin.npz"]
        return names

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        pass

    def process(self):
        data = read_graph(self.raw_dir)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name}()"