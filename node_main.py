import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import os
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
#from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
from utils.utils import N2DataLoader
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
from sklearn.metrics import roc_auc_score
from utils.loss import bcelogits_loss, weighted_cross_entropy, float_bcelogits_loss
from torch_geometric.utils import softmax



# ---- 1. 数据加载 ----
def load_data():
    dataset = TUDataset(root='data/TUDataset', name='IMDB-MULTI')
    return dataset




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
    
'''
class PerNodeFourierGAT(MessagePassing):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__(aggr='add')  
        self.fourier = FourierBaseDynamic(in_dim, out_dim)

        att_in_dim = in_dim + out_dim  
        self.att_mlp = nn.Sequential(
            nn.Linear(att_in_dim, 1),
            nn.LeakyReLU(0.2)
        )


        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index_i):
        msg = self.fourier(h_src=x_j, h_i=x_i)
        att_input = torch.cat([msg, x_i], dim=-1)
        att_score = self.att_mlp(att_input).squeeze(-1)
        att_weight = softmax(att_score, index=edge_index_i)
        return msg * att_weight.view(-1, 1)

    def update(self, aggr_out, x):
        out = self.update_mlp(torch.cat([x, aggr_out], dim=-1))
        return out
    

'''

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

    def update(self, aggr_out, x):
        out = self.update_mlp(torch.cat([x, aggr_out], dim=-1))
        #print(x.shape)
        #print(out.shape)
        return  out  
    

# ---- 3. 图分类模型 ----
'''
class MoleculeGNN_Fourier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_classes, dropout=0.0):
        super().__init__()
        self.conv1 = PerNodeFourierMP(in_dim, hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        # self.conv2 = PerNodeFourierMP(out_dim, hidden_dim, out_dim)
        self.conv2 = PerNodeFourierMP(out_dim, hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        #x = self.bn1(x)
        x = F.relu(x)
        #x = F.silu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        #x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        return self.classifier(x)
'''



class MoleculeGNN_Fourier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_classes, 
                 num_layers=8, dropout=0.0, use_bn=False, use_residual=True, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数选择
        self.act = {
            'relu': F.relu,
            'silu': F.silu,
            'gelu': F.gelu,
        }[act]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # input layer
        self.convs.append(PerNodeFourierMP(in_dim, hidden_dim, out_dim))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(out_dim))

        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(PerNodeFourierMP(out_dim, hidden_dim, out_dim))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(out_dim))

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x_res = x  
            x = conv(x, edge_index)

            if self.use_bn:
                x = self.bns[i](x)
            x = self.act(x)
            x = self.dropout(x)

            if self.use_residual and x.shape == x_res.shape:
                x = x + x_res  

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
def float_bcelogits_loss(pred, label):
    return F.binary_cross_entropy_with_logits(pred, label.float())

def get_loss_func(task_type):
    if task_type in ["multi-class", "binary-class"]:
        return float_bcelogits_loss
    elif task_type == "reg":
        return nn.MSELoss()
    elif "single-class" in task_type:
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported task type: " + task_type)

    
scaler = GradScaler()
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    scaler = GradScaler()
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            out = model(data)

            if hasattr(data, 'train_mask'):
                mask = data.train_mask
                loss = criterion(out[mask], data.y[mask])
            else:
                loss = criterion(out, data.y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if hasattr(data, 'train_mask'):
            total_loss += loss.item() * mask.sum().item()
        else:
            total_loss += loss.item() * data.num_nodes

    total_nodes = sum(
        int(d.train_mask.sum().item()) if hasattr(d, 'train_mask') else d.num_nodes
        for d in loader.dataset
    )
    return total_loss / total_nodes


@torch.no_grad()
def test(model, loader, device, dataset_name):
    model.eval()
    y_true, y_pred = [], []

    for data in loader:
        data = data.to(device)
        with autocast(device_type='cuda'):
            out = model(data)

        if hasattr(data, 'test_mask'):
            mask = data.test_mask
            y_true.append(data.y[mask].cpu())
            y_pred.append(out[mask].detach().cpu())
        else:
            y_true.append(data.y.cpu())
            y_pred.append(out.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)


    acc_datasets = [
        "AmazonComputers", "AmazonPhoto", "CoauthorCS", 
        "CoauthorPhysics", "arxiv-year", "ogbn-arxiv",
        'amazon-ratings'
    ]
    roc_datasets = [
        'minesweeper', 'tolokers', 
        'questions', 'genius', 'ogbn-proteins'
    ]

    if dataset_name in acc_datasets:
        pred = y_pred.argmax(dim=-1)
        correct = (pred == y_true).sum().item()
        return correct / y_true.size(0)

    elif dataset_name in roc_datasets:
        if y_pred.size(1) == 1:
            y_score = y_pred.view(-1).sigmoid().numpy()
            y_true_np = y_true.numpy()
        else:
            y_score = y_pred.sigmoid().numpy()
            y_true_np = F.one_hot(y_true, num_classes=y_pred.size(1)).numpy()
        try:
            auc = roc_auc_score(y_true_np, y_score, average='macro', multi_class='ovr' if y_pred.size(1) > 1 else 'raise')
        except ValueError:
            auc = 0.0
        return auc
    else:
        raise ValueError(f"Unknown dataset for metric: {dataset_name}")


# ---- 5. 主流程 ----
def main_worker(local_rank, args):
    # DDP 初始化
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl')

    
    data_loader = N2DataLoader()
    fold_accuracies = []
    graph_iter_range = 5

    for k in range(graph_iter_range):
        args.fold_idx = k if graph_iter_range > 1 else args.fold_idx
        data_loader.load_data(dataset=args.dataset, spilit_type="public",
                              nbatch=args.nbatch, fold_idx=args.fold_idx)

        nclass, nfeats, nedgefeats = data_loader.nclass, data_loader.nfeats, data_loader.nedgefeats
        metric = data_loader.metric
        task_type = data_loader.task_type

        #print("nedgefeats shape:", nedgefeats.shape)
        train_dataset = data_loader.train_data.dataset
        val_dataset = data_loader.val_data.dataset
        test_dataset = data_loader.test_data.dataset

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=args.nbatch, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.nbatch, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.nbatch, shuffle=False)

        num_classes = nclass
        print("node freature dim is :", nfeats)
        if args.dataset == "minesweeper":
            num_layers = 15
            hidden_dim = 64
            out_dim = 16
            dropout = 0.1
            use_bn = False
        
        model = MoleculeGNN_Fourier(
            in_dim=nfeats,
            hidden_dim=64,
            out_dim=16,
            num_classes=num_classes,
            dropout=0.1
        ).to(device)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = get_loss_func(task_type)

        best_acc = 0
        for epoch in range(1, 501):
            train_loader.sampler.set_epoch(epoch)
            loss = train(model, train_loader, optimizer, criterion, device)
            acc = test(model, test_loader, device, args.dataset)
            if acc > best_acc:
                best_acc = acc
                print(f'the acc of epoch {epoch} is {best_acc}.')
        print(f"\n====== fold {k+1} =====")
        print(f"The best acc is : {best_acc}")

        if rank == 0:
            fold_accuracies.append(best_acc)

    #
    if rank == 0:
        print("\n===== Cross Validation Result =====")
        print(f"Average Accuracy: {sum(fold_accuracies) / len(fold_accuracies):.4f}")
        best_acc = np.array(fold_accuracies)
        mean_acc = best_acc.mean()
        std_acc = best_acc.std()
        print(f"\n[Summary over {len(fold_accuracies)} folds] Best Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="minesweeper")
    parser.add_argument('--fold_idx', type=int, default=1)
    parser.add_argument('--nbatch', type=int, default=1)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    main_worker(local_rank, args)

if __name__ == '__main__':
    # Load data
    run_main()
