import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, Linear
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gdp
from gatedgcn import GatedGCNLayer

class Loop_net(torch.nn.Module):
    def __init__(self, num_classes, embedding_size=70, node_feature_size=1, edge_feature_size=1):
        super(Loop_net, self).__init__()
        self.edge_emb = Linear(edge_feature_size, embedding_size)
        self.node_emb = Linear(node_feature_size, embedding_size)
        self.conv1 = GatedGCNLayer(in_dim=embedding_size,
                                    out_dim=embedding_size,
                                    dropout=0.,  
                                    residual=True,  
                                    equivstable_pe=False)
        self.conv2 = GatedGCNLayer(in_dim=embedding_size,
                                    out_dim=embedding_size,
                                    dropout=0.,  
                                    residual=True,  
                                    equivstable_pe=False)
        self.norm3 = LayerNorm(256, affine=True)
        
        self.classifier1 = Linear(embedding_size*2, 256)
        self.classifier2 = Linear(256, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch_index):
        sizes = torch.nn.functional.one_hot(batch_index.data).sum(dim=0)
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        
        for i in range(int(torch.max(sizes)/2)):
            x, edge_attr = self.conv2(x, edge_index, edge_attr)
            
        x = torch.cat([gmp(x, batch_index),gap(x,batch_index)], axis=1) 
        x = F.relu(self.classifier1(x))
        x = self.norm3(x)
        x = self.classifier2(x)
        return x
    