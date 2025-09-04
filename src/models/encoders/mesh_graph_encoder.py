import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_scatter import scatter_add


def get_activation_fn(activation_fn: str):
    """Get activation module by name for use in nn.Sequential."""
    if activation_fn == 'silu':
        return nn.SiLU()
    elif activation_fn == 'gelu':
        return nn.GELU()
    elif activation_fn == 'relu':
        return nn.ReLU()
    elif activation_fn == 'tanh':
        return nn.Tanh()
    elif activation_fn == 'sigmoid':
        return nn.Sigmoid()
    else:
        return nn.SiLU()  # default



class MLP(nn.Module):
    def __init__(self, input_dim, num_features, num_layers=2, activation_fn="silu", use_bias=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = getattr(F, activation_fn) if hasattr(F, activation_fn) else F.silu

        hidden_dim = num_features
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=use_bias))
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=use_bias))


    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        return self.layers[-1](x)

        
# stub for promote_to_e3x
def promote_to_e3x(x):
    return x


class NodeAttributeEmbedding(nn.Module):
    def __init__(self, input_dim, num_features, activation_fn="silu"):
        super().__init__()
        self.mlp = MLP(input_dim=input_dim, num_layers=2, num_features=num_features, activation_fn=activation_fn)
        self.norm = nn.LayerNorm(num_features)

    def forward(self, batch, **kwargs):
        node_attributes = batch.node_attr  # (num_nodes, node_attr_dim)
        features_nodes = self.mlp(node_attributes)
        features_nodes = self.norm(features_nodes)
        return promote_to_e3x(features_nodes)  # (num_nodes, 1, 1, num_features)


class EdgeAttributeEmbedding(nn.Module):
    def __init__(self, input_dim, num_features, activation_fn="silu"):
        super().__init__()
        self.mlp = MLP(input_dim=input_dim, num_layers=2, num_features=num_features, activation_fn=activation_fn)
        self.norm = nn.LayerNorm(num_features)

    def forward(self, batch, **kwargs):
        edge_attributes = batch.edge_attr  # (num_edges, edge_attr_dim)
        features_edges = self.mlp(edge_attributes)
        features_edges = self.norm(features_edges)
        return promote_to_e3x(features_edges)  # (num_edges, 1, 1, num_features)


class MeshGraphNetLayer(nn.Module):
    def __init__(self, num_edge_features, num_node_features, activation_fn="silu"):
        super().__init__()
        self.num_edge_features = num_edge_features
        self.num_node_features = num_node_features
        self.node_embedding = NodeAttributeEmbedding(input_dim=num_node_features, num_features=num_node_features, activation_fn=activation_fn)
        self.edge_embedding = EdgeAttributeEmbedding(input_dim=num_edge_features, num_features=num_edge_features, activation_fn=activation_fn)

        self.edge_mlp = MLP(input_dim=num_edge_features * 3, num_features=num_edge_features, num_layers=2, activation_fn=activation_fn)
        self.node_mlp = MLP(input_dim=num_node_features * 2, num_features=num_node_features, num_layers=2, activation_fn=activation_fn)

        self.edge_norm = nn.LayerNorm(num_edge_features * 3)  # senders + receivers + edges
        self.node_norm = nn.LayerNorm(num_node_features * 2)
        # node_norm input dim is unknown upfront (nodes + aggregated edges), so we'll apply dynamically

    def forward(self, node_features, edge_features, edge_index):
        # sender/receiver features
        sender_features = node_features[edge_index[0]]
        receiver_features = node_features[edge_index[1]]

        edge_input = torch.cat([sender_features, receiver_features, edge_features], dim=-1)
        edge_features = self.edge_mlp(self.edge_norm(edge_input))

        num_nodes = node_features.size(0)
        agg_edges = scatter_add(edge_features, edge_index[1], dim=0, dim_size=num_nodes)

        node_input = torch.cat([node_features, agg_edges], dim=-1)
        node_features = self.node_mlp(self.node_norm(node_input))

        return node_features, edge_features

class MeshGraphNetEncoder(nn.Module):
    def __init__(self, num_edge_features, num_node_features, num_layers=2, activation_fn="silu"):
        super().__init__()
        self.num_edge_features = num_edge_features
        self.num_node_features = num_node_features
        self.node_embedding = NodeAttributeEmbedding(input_dim=11, num_features=num_node_features, activation_fn=activation_fn)
        self.edge_embedding = EdgeAttributeEmbedding(input_dim=4, num_features=num_edge_features, activation_fn=activation_fn)
        self.layers = nn.ModuleList([MeshGraphNetLayer(num_edge_features, num_node_features, activation_fn=activation_fn) for _ in range(num_layers)])

    def forward(self, batch):
        node_features = self.node_embedding(batch)
        edge_features = self.edge_embedding(batch)
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, batch.edge_index)
        return node_features, edge_features