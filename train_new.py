import os.path as osp
from pickletools import optimize

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling

from model import PHVGNNModel
from utils import *

from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv

class PHVDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.data[index]
        seq_len = len(seq)
        sample = {
            'seq': torch.from_numpy(np.array(seq)),
            'len': torch.tensor(seq_len)
        }
        return sample

def collate_func(batch_dict):
    seq_batch = []
    len_batch = []
    for i in range(len(batch_dict)):
        item = batch_dict[i]
        seq_batch.append(item['seq'])
        len_batch.append(item['len'])

    seq_batch = pad_sequence(seq_batch, batch_first=True)

    return seq_batch, torch.tensor(len_batch)


class PHVEmbedding(nn.Module):
    def __init__(self, max_word, word_embedding_size=32, lstm_hidden_size=32, lstm_layers=3, hidden_size=32):
        super(PHVEmbedding, self).__init__()

        self.embedding_layer = nn.Embedding(max_word, word_embedding_size)
        self.lstm_layer = nn.LSTM(word_embedding_size, lstm_hidden_size, batch_first=True, bidirectional=True, num_layers=lstm_layers)
        self.linear_layer = nn.Linear(lstm_hidden_size * 2, hidden_size)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x, seq_len):
        seq_embedding = self.embedding_layer(x)

        seq_embedding = rnn.pack_padded_sequence(seq_embedding, seq_len, batch_first=True, enforce_sorted=False)

        seq_embedding, _ = self.lstm_layer(seq_embedding)

        seq_lstm_out, lens_unpacked = rnn.pad_packed_sequence(seq_embedding, batch_first=True)

        seq_lstm_out = torch.cat([seq_lstm_out[i:i+1, lens_unpacked[i]-1, :] for i in range(lens_unpacked.size()[0])], dim=0)

        seq_lstm_out = seq_lstm_out.view(seq_lstm_out.size()[0], -1)

        out = self.linear_layer(seq_lstm_out)

        return out

class PHVClassifier(nn.Module):
    def __init__(self, hidden_size=32):
        super(PHVClassifier, self).__init__()

        self.conv1 = GCNConv(hidden_size, 128)
        self.conv2 = GCNConv(128, hidden_size)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x, pos_edge_index, neg_edge_index):

        seq_embedded = F.relu(self.conv1(x, pos_edge_index))
        seq_embedded = F.relu(self.conv2(seq_embedded, pos_edge_index))

        edge_index = torch.cat([pos_edge_index,neg_edge_index], dim=-1)

        out = torch.cat([seq_embedded[edge_index[0]], seq_embedded[edge_index[1]]], dim=1)
        out = self.output_layer(out)
        return out

    def inference(self, x, edge_index, test_edge_index):
        seq_embedded = F.relu(self.conv1(x, edge_index))
        seq_embedded = F.relu(self.conv2(seq_embedded, edge_index))

        out = torch.cat([seq_embedded[test_edge_index[0]], seq_embedded[test_edge_index[1]]], dim=1)
        out = self.output_layer(out)
        return out

folder = 'data'
node_file = osp.join(folder, 'all_seq.txt')
edge_file = osp.join(folder, 'all_edge.txt')
train_edge_file = osp.join(folder, 'train_edge.txt')
test_edge_file = osp.join(folder, 'test_edge.txt')

node_list = []
with open(node_file, 'r') as f:
    for line in f:
        line = line.strip()
        features = line.split(' ')
        features = [int(i) for i in features]
        node_list.append(features)

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

node_dataset = PHVDataset(node_list)
node_dataloader = DataLoader(node_dataset, batch_size=32, shuffle=True, collate_fn=collate_func)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
e_model = PHVEmbedding(max_word=num_word(), word_embedding_size=32, lstm_hidden_size=32, lstm_layers=3, hidden_size=32).to(device)
c_model = PHVClassifier(hidden_size=32).to(device)
optimizor = torch.optim.Adam(list(e_model.parameters()) + list(c_model.parameters()), lr=0.01)

for epoch in range(100):
    e_model.train()
    c_model.train()

    optimizor.zero_grad()

    node_features = []
    for i, data in enumerate(node_dataloader):
        x, x_len = data
        x = x.to(device)
        features = e_model(x, x_len)
        node_features.append(features)
        torch.cuda.empty_cache()

    
    node_features = torch.cat(node_features, dim=0)
    pos_edge_index = train_edge_index.to(device)
    neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=node_features.size(0),
            num_neg_samples=pos_edge_index.size(1)).to(device)
    pred = c_model(node_features, pos_edge_index, neg_edge_index)
    loss = F.cross_entropy(pred, torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(device).long())
    
    loss.backward()
    optimizor.step()

    print('epoch: {}, loss: {}'.format(epoch, loss.item()))

    with torch.no_grad():
        e_model.eval()
        c_model.eval()

        node_features = []
        for i, data in enumerate(node_dataloader):
            x, x_len = data
            x = x.to(device)
            features = e_model(x, x_len)
            node_features.append(features)
            torch.cuda.empty_cache()

        node_features = torch.cat(node_features, dim=0)
        pred = c_model.inference(node_features, pos_edge_index, test_edge_index)
        pred = torch.argmax(pred, dim=1).to('cpu')
        acc = (pred == torch.ones(test_edge_index.size(1))).sum().item() / test_edge_index.size(1)
        print('epoch: {}, acc: {}'.format(epoch, acc))

    




