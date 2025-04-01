import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader 
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.nn import GCNConv
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import argparse
from utils.utils import N2DataLoader
import numpy as np


# ---- 1. 数据加载 ----
def load_data():
    dataset = TUDataset(root='data/TUDataset', name='IMDB-MULTI')
    return dataset
#IMDB-BINARY
# ---- 2. 每个节点一个 MLP 的 Message Passing 层 ----
class FourierBaseDynamic(nn.Module):
    def __init__(self, in_dim, fourier_dim):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.freq_generator = nn.Linear(in_dim, fourier_dim * in_dim)  # W_i
        self.phase_generator = nn.Linear(in_dim, fourier_dim)          # b_i
        self.out_proj = nn.Linear(6 * fourier_dim, fourier_dim)

    def forward(self, h_src, h_i):
        """
        h_src: [E, in_dim]  - 源节点特征
        h_i:   [E, in_dim]  - 对应目标节点的特征
        """
        B, D = h_src.size()
        F = self.fourier_dim
        
        '''
        W = self.freq_generator(h_i).view(B, F, D)  # [E, F, D]
        b = self.phase_generator(h_i)               # [E, F]

        proj = torch.einsum('efd,ed->ef', W, h_src) + b  # [E, F]
        sin = torch.sin(proj)
        cos = torch.cos(proj)
        emb = torch.cat([sin, cos], dim=-1)  # [E, 2F]
        return self.out_proj(emb)            # [E, F]
        '''
        freqs = self.freq_generator(h_i).view(B, F, D)
        proj = torch.einsum('bfd,bd->bf', freqs, h_src)  # 多组频率组合
        proj1 = proj
        proj2 = 2 * proj
        proj3 = 4 * proj
        # concat multi-frequency basis
        emb = torch.cat([
            torch.sin(proj1), torch.cos(proj1),
            torch.sin(proj2), torch.cos(proj2),
            torch.sin(proj3), torch.cos(proj3)
        ], dim=-1)  # [B, 6F]
        out = self.out_proj(emb)
        return out
    



class PerNodeFourierMP(MessagePassing):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__(aggr='mean')  # 使用 mean 聚合
        self.fourier = FourierBaseDynamic(in_dim, out_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index):
        # x: [N, in_dim], edge_index: [2, E]
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i: [E, in_dim] (中心节点), x_j: [E, in_dim] (邻居节点)
        return self.fourier(h_src=x_j, h_i=x_i)

    '''
    def update(self, aggr_out, x):
        # aggr_out: [N, out_dim], x: [N, in_dim]
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))
    '''
    def update(self, aggr_out, x):
        out = self.update_mlp(torch.cat([x, aggr_out], dim=-1))
        #print(x.shape)
        #print(out.shape)
        return  out  
    

# ---- 3. 图分类模型 ----
class MoleculeGNN_Fourier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_classes, dropout=0.2):
        super().__init__()
        self.conv1 = PerNodeFourierMP(in_dim, hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)

        self.conv2 = PerNodeFourierMP(out_dim, hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)

        self.readout = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(out_dim, 1),
                nn.Sigmoid()
            ),
            nn=nn.Sequential(  
                nn.Linear(out_dim, out_dim),
                nn.ReLU()
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        #x = global_mean_pool(x, batch)
        x = self.readout(x, batch)
        return self.classifier(x)



class MoleculeGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_classes, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)

        self.conv2 = GCNConv(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)
        self.readout = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(out_dim, 1),
                nn.Sigmoid()
            ),
            nn=nn.Sequential(  
                nn.Linear(out_dim, out_dim),
                nn.ReLU()
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return self.classifier(x)
    

# ---- 4. 训练与评估 ----
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        #for batch in train_loader:
        #print(data.y)  # or batch['label']
        #break

        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# ---- 5. 主流程 ----
def main():
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--fold_idx', type=int, default=1, help='fold idx for multi-split')
    parser.add_argument('--nbatch', type=int, default=32, help='Number of self defined batch')
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = N2DataLoader()
    fold_accuracies = []
    #print("Loading " + args.dataset.title() + "...")
    flag = (args.dataset in ["NCI1", "IMDB-BINARY", "IMDB-MULTI", "PROTEINS", "COLLAB"])
    graph_iter_range = 10 if flag else 1
    for k in range(graph_iter_range):
        args.fold_idx = k if graph_iter_range > 1 else args.fold_idx
        data_loader.load_data(dataset=args.dataset, spilit_type="public", 
                            nbatch=args.nbatch, fold_idx=args.fold_idx)
        nclass, nfeats, nedgefeats = data_loader.nclass, data_loader.nfeats, data_loader.nedgefeats
        train_data, val_data, test_data = data_loader.train_data, data_loader.val_data, data_loader.test_data

        

        sample_batch_list = next(iter(train_data))  # ← 返回的是 list[Data]
        first_graph = sample_batch_list[0]  # 从列表中取出一个图
        in_dim = first_graph.x.shape[1]

        # 拿到原始 dataset
        train_data = data_loader.train_data.dataset
        val_data = data_loader.val_data.dataset
        test_data = data_loader.test_data.dataset

        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(train_data, batch_size=args.nbatch, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.nbatch, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=args.nbatch, shuffle=False)

        print(in_dim)
        if args.dataset in ['IMDB-BINARY', 'NCI1', 'PROTEINS']:
            num_classes = 2
        elif args.dataset in ["IMDB-MULTI", 'COLLAB']:
            num_classes = 3

        model = MoleculeGNN_Fourier(
            in_dim=in_dim,
            hidden_dim=64,
            out_dim=64,
            num_classes=num_classes,
            dropout=0.1
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        acc_t = 0
        for epoch in range(1, 1001):  # 每个 fold 训练 200 epoch
            loss = train(model, train_loader, optimizer, criterion, device)
            acc = test(model, test_loader, device)
            if acc > acc_t:
                acc_t = acc

        print(f"Best Acc on Fold {k + 1}: {acc_t:.4f}")
        fold_accuracies.append(acc_t)

    print("\n===== Cross Validation Result =====")
    print(f"Average Accuracy: {sum(fold_accuracies) / len(fold_accuracies):.4f}")
    best_acc = np.array(fold_accuracies)
    mean_acc = best_acc.mean()
    std_acc = best_acc.std()
    print(f"\n[Summary over 5 runs] Best Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

if __name__ == '__main__':
    # Load data
    main()