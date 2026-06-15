# rgcn_distmult.py

from typing import Dict, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import FastRGCNConv


###############################################################################
# Utilities
###############################################################################

def hetero_to_rgcn_format(
    data: HeteroData,
    edge_types_to_use,
):
    """
    Convert HeteroData edge_index_dict into:
        edge_index : [2, num_edges]
        edge_type  : [num_edges]

    suitable for RGCNConv / FastRGCNConv.

    Assumes:
        - all node types share the same node ID space
        - one global homogeneous graph

    Returns:
        edge_index
        edge_type
        relation_to_id
    """

    edge_indices = []
    edge_types = []

    relation_to_id = {
        rel: i
        for i, rel in enumerate(edge_types_to_use)
    }

    for rel, rel_id in relation_to_id.items():

        if rel not in data.edge_index_dict:
            continue

        ei = data[rel].edge_index

        edge_indices.append(ei)

        edge_types.append(
            th.full(
                (ei.size(1),),
                rel_id,
                dtype=th.long,
                device=ei.device,
            )
        )

    edge_index = th.cat(edge_indices, dim=1)
    edge_type = th.cat(edge_types, dim=0)

    return edge_index, edge_type, relation_to_id


###############################################################################
# Encoder
###############################################################################

class RGCNEncoder(nn.Module):

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embedding_dim: int,
        hidden_channels: int,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.node_emb = nn.Embedding(
            num_nodes,
            embedding_dim,
        )

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            FastRGCNConv(
                embedding_dim,
                hidden_channels,
                num_relations=num_relations,
                num_bases=num_bases,
            )
        )

        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(
                FastRGCNConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations=num_relations,
                    num_bases=num_bases,
                )
            )

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.node_emb.weight)

        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        node_ids: th.Tensor,
        edge_index: th.Tensor,
        edge_type: th.Tensor,
    ):

        x = self.node_emb(node_ids)

        for layer in self.layers[:-1]:

            x = layer(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.layers[-1](x, edge_index, edge_type)

        return x


###############################################################################
# DistMult Decoder
###############################################################################

class DistMultDecoder(nn.Module):

    def __init__(
        self,
        num_relations: int,
        hidden_channels: int,
    ):
        super().__init__()

        self.rel_emb = nn.Embedding(
            num_relations,
            hidden_channels,
        )

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(
        self,
        z: th.Tensor,
        edge_label_index: th.Tensor,
        rel_ids: th.Tensor,
    ):

        src, dst = edge_label_index

        z_src = z[src]
        z_dst = z[dst]

        r = self.rel_emb(rel_ids)

        scores = (z_src * r * z_dst).sum(dim=-1)

        return scores


###############################################################################
# Full RGCN + DistMult Model
###############################################################################

class RGCNDistMult(nn.Module):

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        hidden_channels: int,
        message_edge_types,
        predicted_edge_types=None,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout: float = 0.2,
        device: str = 'cpu',
    ):
        super().__init__()

        if predicted_edge_types is None:
            predicted_edge_types = message_edge_types

        self.device = device

        self.message_edge_types = list(message_edge_types)
        self.predicted_edge_types = list(predicted_edge_types)

        self.relation_to_id = {
            rel: i
            for i, rel in enumerate(self.message_edge_types)
        }

        self.encoder = RGCNEncoder(
            num_nodes=num_nodes,
            num_relations=len(self.relation_to_id),
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_bases=num_bases,
            dropout=dropout,
        )

        self.decoder = DistMultDecoder(
            num_relations=len(self.relation_to_id),
            hidden_channels=hidden_channels,
        )

    @property
    def neighbors_to_sample(self):
        return None

    def set_neighbors_to_sample(
        self,
        neighbors,
        val_neighbors=None,
    ):
        # compatibility with existing training code
        pass

    def encode(
        self,
        data: HeteroData,
    ):

        edge_index, edge_type, _ = hetero_to_rgcn_format(
            data,
            self.message_edge_types,
        )

        # assumes single shared node domain
        node_type = data.node_types[0]

        node_ids = data[node_type].node_id

        z = self.encoder(
            node_ids=node_ids,
            edge_index=edge_index,
            edge_type=edge_type,
        )

        return z

    def decode(
        self,
        z: th.Tensor,
        edges_to_predict: Dict[Tuple[str, str, str], th.Tensor],
    ):

        all_scores = []

        for rel, edge_label_index in edges_to_predict.items():

            rel_id = self.relation_to_id[rel]

            rel_ids = th.full(
                (edge_label_index.size(1),),
                rel_id,
                dtype=th.long,
                device=self.device,
            )

            scores = self.decoder(
                z=z,
                edge_label_index=edge_label_index,
                rel_ids=rel_ids,
            )

            all_scores.append(scores)

        return th.cat(all_scores, dim=0)

    def forward(
        self,
        data: HeteroData,
        edges_to_predict=None,
        return_embs: bool = False,
    ):

        if edges_to_predict is None:
            edges_to_predict = data.edge_label_index_dict

        z = self.encode(data)

        scores = self.decode(
            z,
            edges_to_predict,
        )

        if return_embs:
            return scores, z
        else:
            return scores
        

class SingleRGCNDistMult(nn.Module):

    def __init__(
        self,
        num_nodes,
        embedding_dim,
        hidden_dim,
        edge_types,
        num_layers=2,
        num_bases=30,
        dropout=0.2,
    ):
        super().__init__()

        self.relation_to_id = {
            rel: i
            for i, rel in enumerate(edge_types)
        }

        self.node_emb = nn.Embedding(
            num_nodes,
            embedding_dim,
        )

        self.convs = nn.ModuleList()

        self.convs.append(
            FastRGCNConv(
                embedding_dim,
                hidden_dim,
                num_relations=len(edge_types),
                num_bases=num_bases,
            )
        )

        for _ in range(num_layers - 1):

            self.convs.append(
                FastRGCNConv(
                    hidden_dim,
                    hidden_dim,
                    num_relations=len(edge_types),
                    num_bases=num_bases,
                )
            )

        # SINGLE target relation embedding
        self.target_rel = nn.Parameter(
            th.randn(hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.node_emb.weight)

        for conv in self.convs:
            conv.reset_parameters()

        nn.init.xavier_uniform_(
            self.target_rel.unsqueeze(0)
        )

    def build_rgcn_inputs(self, data):

        edge_indices = []
        edge_types = []

        for rel, rel_id in self.relation_to_id.items():

            ei = data[rel].edge_index

            edge_indices.append(ei)

            edge_types.append(
                th.full(
                    (ei.size(1),),
                    rel_id,
                    dtype=th.long,
                    device=ei.device,
                )
            )

        edge_index = th.cat(edge_indices, dim=1)
        edge_type = th.cat(edge_types)

        return edge_index, edge_type

    def encode(self, data, return_embs=False):

        edge_index, edge_type = self.build_rgcn_inputs(data)

        node_type = data.node_types[0]

        x = self.node_emb(
            data[node_type].node_id
        )

        if return_embs:
            embs = [{node_type: x}]

        for conv in self.convs[:-1]:

            x = conv(x, edge_index, edge_type)
            if return_embs:
                embs.append({node_type: x})
            x = F.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](
            x,
            edge_index,
            edge_type,
        )
        if return_embs:
            embs.append({node_type: x})

        if return_embs:
            return x, embs
        return x

    def decode(
        self,
        z,
        edge_label_index,
    ):

        src, dst = edge_label_index

        z_src = z[src]
        z_dst = z[dst]

        return (
            z_src
            * self.target_rel
            * z_dst
        ).sum(dim=-1)

    def forward(
        self,
        data,
        edge_label_index,
        return_embs=False,
    ):

        if return_embs:
            z, x_dicts = self.encode(data, return_embs=True)    
        z = self.encode(data)

        scores = self.decode(
            z,
            edge_label_index,
        )

        if return_embs:
            return scores, x_dicts

        return scores


class SingleRGCNComplEx(nn.Module):

    def __init__(
        self,
        num_nodes,
        embedding_dim,
        hidden_dim,
        edge_types,
        num_layers=2,
        num_bases=30,
        dropout=0.2,
    ):
        super().__init__()

        assert hidden_dim % 2 == 0, \
            "hidden_dim must be even for ComplEx"

        self.hidden_dim = hidden_dim
        self.complex_dim = hidden_dim // 2

        self.relation_to_id = {
            rel: i
            for i, rel in enumerate(edge_types)
        }

        self.node_emb = nn.Embedding(
            num_nodes,
            embedding_dim,
        )

        self.convs = nn.ModuleList()

        self.convs.append(
            FastRGCNConv(
                embedding_dim,
                hidden_dim,
                num_relations=len(edge_types),
                num_bases=num_bases,
            )
        )

        for _ in range(num_layers - 1):

            self.convs.append(
                FastRGCNConv(
                    hidden_dim,
                    hidden_dim,
                    num_relations=len(edge_types),
                    num_bases=num_bases,
                )
            )

        # SINGLE complex relation embedding
        self.target_rel = nn.Parameter(
            th.randn(hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.node_emb.weight)

        for conv in self.convs:
            conv.reset_parameters()

        nn.init.xavier_uniform_(
            self.target_rel.unsqueeze(0)
        )

    def build_rgcn_inputs(self, data):

        edge_indices = []
        edge_types = []

        for rel, rel_id in self.relation_to_id.items():

            ei = data[rel].edge_index

            edge_indices.append(ei)

            edge_types.append(
                th.full(
                    (ei.size(1),),
                    rel_id,
                    dtype=th.long,
                    device=ei.device,
                )
            )

        edge_index = th.cat(edge_indices, dim=1)
        edge_type = th.cat(edge_types)

        return edge_index, edge_type

    def encode(self, data, return_embs=False):

        edge_index, edge_type = self.build_rgcn_inputs(data)

        node_type = data.node_types[0]

        x = self.node_emb(
            data[node_type].node_id
        )

        if return_embs:
            embs = [{node_type: x}]

        for conv in self.convs[:-1]:

            x = conv(x, edge_index, edge_type)

            if return_embs:
                embs.append({node_type: x})

            x = F.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](
            x,
            edge_index,
            edge_type,
        )

        if return_embs:
            embs.append({node_type: x})

        if return_embs:
            return x, embs

        return x

    def split_complex(self, x):

        return (
            x[..., :self.complex_dim],
            x[..., self.complex_dim:],
        )

    def decode(
        self,
        z,
        edge_label_index,
    ):

        src, dst = edge_label_index

        h = z[src]
        t = z[dst]

        h_re, h_im = self.split_complex(h)
        t_re, t_im = self.split_complex(t)

        r_re, r_im = self.split_complex(
            self.target_rel
        )

        scores = (
            h_re * r_re * t_re
            + h_im * r_re * t_im
            + h_re * r_im * t_im
            - h_im * r_im * t_re
        ).sum(dim=-1)

        return scores

    def forward(
        self,
        data,
        edge_label_index,
        return_embs=False,
    ):

        if return_embs:
            z, x_dicts = self.encode(
                data,
                return_embs=True
            )
        else:
            z = self.encode(data)

        scores = self.decode(
            z,
            edge_label_index,
        )

        if return_embs:
            return scores, x_dicts

        return scores
    

class SingleRGCNNN(nn.Module):

    def __init__(
        self,
        num_nodes,
        embedding_dim,
        hidden_dim,
        nn_layers,
        edge_types,
        num_layers=2,
        num_bases=30,
        dropout=0.0,
    ):
        super().__init__()

        assert hidden_dim % 2 == 0, \
            "hidden_dim must be even for ComplEx"

        self.hidden_dim = hidden_dim
        self.complex_dim = hidden_dim // 2

        self.relation_to_id = {
            rel: i
            for i, rel in enumerate(edge_types)
        }

        self.node_emb = nn.Embedding(
            num_nodes,
            embedding_dim,
        )

        self.convs = nn.ModuleList()

        self.convs.append(
            FastRGCNConv(
                embedding_dim,
                hidden_dim,
                num_relations=len(edge_types),
                num_bases=num_bases,
            )
        )

        for _ in range(num_layers - 1):

            self.convs.append(
                FastRGCNConv(
                    hidden_dim,
                    hidden_dim,
                    num_relations=len(edge_types),
                    num_bases=num_bases,
                )
            )

        # # SINGLE complex relation embedding

        prev = hidden_dim*2
        self.lin = nn.ModuleList()
        for l in nn_layers:
            self.lin.append(nn.Linear(prev, l))
            prev = l

        self.final = nn.Linear(prev, 1)

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.node_emb.weight)

        for conv in self.convs:
            conv.reset_parameters()

        for lin in self.lin:
            lin.reset_parameters()

        self.final.reset_parameters()


    def build_rgcn_inputs(self, data):

        edge_indices = []
        edge_types = []

        for rel, rel_id in self.relation_to_id.items():

            ei = data[rel].edge_index

            edge_indices.append(ei)

            edge_types.append(
                th.full(
                    (ei.size(1),),
                    rel_id,
                    dtype=th.long,
                    device=ei.device,
                )
            )

        edge_index = th.cat(edge_indices, dim=1)
        edge_type = th.cat(edge_types)

        return edge_index, edge_type

    def encode(self, data, return_embs=False):

        edge_index, edge_type = self.build_rgcn_inputs(data)

        node_type = data.node_types[0]

        x = self.node_emb(
            data[node_type].node_id
        )

        if return_embs:
            embs = [{node_type: x}]

        for conv in self.convs[:-1]:

            x = conv(x, edge_index, edge_type)

            if return_embs:
                embs.append({node_type: x})

            x = F.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](
            x,
            edge_index,
            edge_type,
        )

        if return_embs:
            embs.append({node_type: x})

        if return_embs:
            return x, embs

        return x

    def split_complex(self, x):

        return (
            x[..., :self.complex_dim],
            x[..., self.complex_dim:],
        )

    def decode(
        self,
        z,
        edge_label_index,
    ):

        src, dst = edge_label_index

        h = z[src]
        t = z[dst]

        link = th.cat((h,t), dim=-1)

        for lin in self.lin:
            link = lin(link)
            link = F.relu(link)
            link = self.dropout(link)

        scores = self.final(link)
        return scores

    def forward(
        self,
        data,
        edge_label_index,
        return_embs=False,
    ):

        if return_embs:
            z, x_dicts = self.encode(
                data,
                return_embs=True
            )
        else:
            z = self.encode(data)

        scores = self.decode(
            z,
            edge_label_index,
        )

        if return_embs:
            return scores, x_dicts

        return scores