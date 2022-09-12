import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class Net(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Net,self).__init__()
		self.conv1 = GCNConv(in_channels, 128)
		self.conv2 = GCNConv(128, out_channels)

	def encode(self, x, edge_index): # 节点表征学习
		x = self.conv1(x,edge_index)
		x = x.relu()
		x = self.conv2(x,edge_index)
		return x

	def decode(self, z, pos_edge_index, neg_edge_index): # z传入经过表征学习的所有节点特征矩阵
		edge_index = torch.cat([pos_edge_index,neg_edge_index], dim=-1) # dim=-1, 2维就是1
		return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1) # 头尾节点属性对应相乘后求和
		# 返回一个 [(正样本数+负样本数),1] 的向量

	def decode_all(self,z):
		prob_adj = z @ z.t() # 头节点属性和尾节点属性对应相乘后求和，[节点数，节点数]
		return (prob_adj > 0).nonzero(as_tuple=False).t() # [2,m], 列存储有边的nodes的序号

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset) 
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
dataset = Planetoid('dataset','Cora',transform=T.NormalizeFeatures())
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
print(data)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, 64).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)



# 训练
def train(data):
	model.train()
	print(data.__dict__.keys())

	# 负采样
	neg_edge_index = negative_sampling(edge_index = data.train_pos_edge_index, # 使得该函数只对训练集中不存在边的节点采样
										num_nodes = data.num_nodes,
										num_neg_samples = data.train_pos_edge_index.size(1))
	


	optimizer.zero_grad()
	
	# 节点表征学习
	z = model.encode(data.x, data.train_pos_edge_index) 
	# 有无边的概率计算
	link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
	# 真实边情况[0,1]，调用get_link_labels
	link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(device)
	# 损失计算
	loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
	# 反向求导
	loss.backward()
	# 迭代
	optimizer.step()

	return loss

# 生成正负样本边的标记
def get_link_labels(pos_edge_index, neg_edge_index):
	num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
	link_labels = torch.zeros(num_links,dtype=torch.float) # 向量
	link_labels[:pos_edge_index.size(1)] = 1
	return link_labels
									

# 测试
@torch.no_grad()
def test(data):
    model.eval()

	# 计算所有的节点表征
    z = model.encode(data.x, data.train_pos_edge_index)

    results = []
    for prefix in ['val','test']:
        # 正负edge_index
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        # 有无边的概率预测
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        # 真实情况
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        # 存入准确率
        results.append(roc_auc_score(link_labels.cpu(),link_probs.cpu()))
        
    return results


# 训练验证与测试
best_val_auc = test_auc = 0
for epoch in range(1, 101):
    loss = train(data)
    val_auc, tmp_test_auc = test(data) # 训练一次计算一次验证、测试准确率
    if val_auc > best_val_auc:
        best_val = val_auc
        test_auc = tmp_test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}') # 03d，不足3位前面补0，大于3位照常输出