# %%
import mowl
mowl.init_jvm("8g")

from mowl.datasets.el import ELDataset
from mowl.datasets.base import PathDataset
import os
import copy
import pickle
import torch
from torch_geometric.data import HeteroData
torch.manual_seed(0)
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
# %%
def sample_neg_ex(num_pos, max_h, max_t, neg_ratio=10, device='cpu',
                  head=None, tail=None):
    """
    Negative sampling for KG edges.

    Args:
        num_pos: number of positive edges
        max_h: number of head nodes
        max_t: number of tail nodes
        neg_ratio: negatives per positive
        head: optional tensor of heads to FIX (shape [1, num_neg] or [1, num_pos])
        tail: optional tensor of tails to FIX (same logic)

    Returns:
        Tensor of shape [2, num_neg]
    """
    num_neg = neg_ratio * num_pos
    # Normalize shape → [1, N]
    if head is not None:
        head = head.to(device)
        if head.dim() == 1:
            head = head.unsqueeze(0)
        if head.shape[1] == num_pos:
            head = head.repeat_interleave(neg_ratio, dim=1)
        heads = head
    else:
        heads = torch.randint(0, max_h, (1, num_neg), device=device)

    if tail is not None:
        tail = tail.to(device)
        if tail.dim() == 1:
            tail = tail.unsqueeze(0)
        if tail.shape[1] == num_pos:
            tail = tail.repeat_interleave(neg_ratio, dim=1)
        tails = tail
    else:
        tails = torch.randint(0, max_t, (1, num_neg), device=device)

    return torch.cat([heads, tails], dim=0)


# %%
data = PathDataset(os.path.join(BASE, 'datafiles/aifbfixed_complete.n3'))
el_dataset = ELDataset(data.ontology)
el_dataset.load()
# %%
gci_dict = {k: v.data for k,v in el_dataset.get_gci_datasets().items() if len(v.data) > 0}
# %%
gci0 = gci_dict['class_assertion'].flip(1)
# %%
class_assertion_offset = (gci0[:,0].max() + 1).item()
gci0[:,1] = gci0[:,1] + class_assertion_offset
class_assertion_len = len(gci0)
print(class_assertion_offset)
print(class_assertion_len)
# %%
gci0 = torch.concat((gci0, gci_dict['gci0']+class_assertion_offset), dim=0)
# %%
rels = el_dataset.object_property_index_dict
rev_rels = {v:k.split('#')[-1] for k,v in rels.items()}
# %%
graph = HeteroData()
graph['c'].node_id = gci0.unique()

# %%
links = gci_dict['object_property_assertion']
print(links[:,1].unique())
# %%
for k,v in rev_rels.items():
    graph['c', v, 'c'].edge_index = links[links[:,1] == k][:,[0,2]].T


# %%
predicted_edge_types = [('c', 'publication', 'c')]
train_graph = copy.copy(graph)
valid_graph = copy.copy(graph)
test_graph = copy.copy(graph)
# %%
VAL_RATIO = 0.1
TEST_RATIO = 0.1
NEG_VAL_RATIO = 500

for e in predicted_edge_types:
    edge_index = graph[e].edge_index
    num_edges = edge_index.shape[1]
    perm = torch.randperm(num_edges)
    edge_index = edge_index[:, perm]
    val_size = int(num_edges * VAL_RATIO)
    test_size = int(num_edges * TEST_RATIO)
    train_size = num_edges - val_size - test_size
    train_edges = edge_index[:, :train_size]
    val_edges = edge_index[:, train_size:train_size+val_size]
    test_edges = edge_index[:, train_size+val_size:]
    
    train_graph[e].edge_index = train_edges

    valid_graph[e].edge_index = train_edges
    valid_graph[e].pos_edge_label_index = val_edges
    neg = torch.cat((sample_neg_ex(val_edges.shape[1], class_assertion_offset,
                                   class_assertion_offset, head=val_edges[0,:],
                                   neg_ratio=NEG_VAL_RATIO), 
                     sample_neg_ex(val_edges.shape[1], class_assertion_offset,
                                   class_assertion_offset, tail=val_edges[1,:],
                                   neg_ratio=NEG_VAL_RATIO)), dim=-1)
    valid_graph[e].neg_edge_label_index = neg

    test_message_edges = torch.cat((train_edges, val_edges), dim=1)
    test_graph[e].edge_index = test_message_edges
    test_graph[e].pos_edge_label_index = test_edges
    neg = torch.cat((sample_neg_ex(test_edges.shape[1], class_assertion_offset,
                                   class_assertion_offset, head=test_edges[0,:],
                                   neg_ratio=NEG_VAL_RATIO), 
                     sample_neg_ex(test_edges.shape[1], class_assertion_offset,
                                   class_assertion_offset, tail=test_edges[1,:],
                                   neg_ratio=NEG_VAL_RATIO)), dim=-1)
    test_graph[e].neg_edge_label_index = neg
# %%
# add this later? to avoid data leakage?
for e, d in train_graph.edge_index_dict.items():
    train_graph[(e[2], f'rev_{e[1]}', e[0])].edge_index = d.flip(0)
    valid_graph[(e[2], f'rev_{e[1]}', e[0])].edge_index = d.flip(0)

for e, d in test_graph.edge_index_dict.items():
    test_graph[(e[2], f'rev_{e[1]}', e[0])].edge_index = d.flip(0)
# %%
with open(os.path.join(BASE, 'dataset/aifb/aifb_graph.pkl'), 'wb') as fo:
    # pickle.dump({'graph': graph, 'gci0': {'c': gci0}, 'max_id': class_assertion_offset}, fo)
    pickle.dump({'graph': graph, 'train': train_graph, 'valid': valid_graph, 'test': test_graph, 'gci0': {'c': gci0}, 'max_id': class_assertion_offset}, fo)
# %%
