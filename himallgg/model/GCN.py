import torch.nn as nn
from torch_geometric.nn import RGCNConv, GraphConv, GATConv, TransformerConv, GCN2Conv, GAT


class GCN(nn.Module):

    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GCN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        # self.num_relations = 4
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=10)
        self.conv2 = GraphConv(h1_dim, h2_dim)


    def forward(self, node_features, edge_index, edge_norm, edge_type):
        # x = self.conv1(node_features, edge_index, edge_type, edge_norm=edge_norm)
        x = self.conv1(node_features, edge_index, edge_type)
        x = self.conv2(x, edge_index)


        return x
    
class SGCN(nn.Module):

    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(SGCN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        self.conv1 = TransformerConv(g_dim, h2_dim)


    def forward(self, node_features, edge_index, edge_norm, edge_type):
        # x = self.conv1(node_features, edge_index, edge_type, edge_norm=edge_norm)
        x = self.conv1(node_features, edge_index)
        
 

        return x






















