
import numpy as np
import torch
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

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

