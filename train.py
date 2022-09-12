import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.loader import DataLoader

from dataset import GraphDataset
from model import PHVGNNModel
from utils import *

dataset = GraphDataset('data', 'PHV')
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
# data = train_test_split_edges(data)
print(data)






device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PHVGNNModel(max_word=num_word()).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)



# 训练
def train(data):
    model.train()
    loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pos_edge_index = batch.edge_index
        # 生成负样本
        neg_edge_index = negative_sampling(
            edge_index=data.pos_edge_index, num_nodes=batch.num_nodes,
            num_neg_samples=pos_edge_index.size(1))
        # 生成正负样本的标签
        link_labels = get_link_labels(pos_edge_index, neg_edge_index).to(device)
        # 计算预测的边的得分
        link_logits = model(data, pos_edge_index, neg_edge_index)
        # 计算loss
        link_loss = F.cross_entropy(link_logits, link_labels)
        link_loss.backward()
        optimizer.step()

        loss += link_loss.item()

    return loss / len(train_loader)

# 生成正负样本边的标记
def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float) # 向量
    link_labels[:pos_edge_index.size(1)] = 1
    return link_labels

# 测试
@torch.no_grad()
def test(data):
    model.eval()
    neg_edge_index = negative_sampling(edge_index = data.edge_index, # 使得该函数只对训练集中不存在边的节点采样
                                        num_nodes = data.num_nodes,
                                        num_neg_samples = data.test_edge_index.size(1))

    link_logits = model(data.x, data.test_edge_index, neg_edge_index, data.seq_len)
    link_labels = get_link_labels(data.test_edge_index, neg_edge_index).to(device)

    return roc_auc_score(link_labels.cpu(), link_logits.cpu())

best_val_auc = test_auc = 0
for epoch in range(1, 101):
    loss = train(data)
    test_auc = test(data) # 训练一次计算一次验证、测试准确率
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f} Test: {test_auc:.4f}')
