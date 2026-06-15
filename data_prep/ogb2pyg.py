# %%

from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
import os, json, pickle
import torch
import tqdm
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)

# %%
dataset = PygLinkPropPredDataset(name = "ogbl-biokg", root = os.path.join(BASE,'dataset/'))
split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
graph = dataset[0]

# %%
all_index = {}
gci0_dir = os.path.join(BASE, 'biokg_gci0')
for fname in os.listdir(gci0_dir):
    if fname.startswith('class_index'):
        domain = fname.split('_')[-1].split('.')[0]
        fpath = os.path.join(gci0_dir, fname)
        with open(fpath, 'r') as fi:
            all_index[domain] = {int(k): v for k,v in json.load(fi).items()}
print(all_index)

# %%
rel_map = {}
f = os.path.join(BASE, 'dataset/ogbl_biokg/mapping/relidx2relname.csv')
with open(f, 'r') as relfile:
    next(relfile)  # skip header
    for line in relfile:
        idx, name = line.strip().split(',', 1)
        rel_map[int(idx)] = name
print(rel_map)

# %%

edge_index_dict = {}
for i in tqdm.tqdm(range(len(train_edge['head']))):
    dom, ran = train_edge['head_type'][i], train_edge['tail_type'][i]
    h, t = train_edge['head'][i].item(), train_edge['tail'][i].item()
    r = rel_map[train_edge['relation'][i].item()]
    key = (dom, r, ran)
    if key in edge_index_dict:
        edge_index_dict[key].append((h, t))
    else:
        edge_index_dict[key] = [(h, t)]
# %%
pyg_data = HeteroData()
for k,v in all_index.items():
    pyg_data[k].node_id = torch.arange(len(v), dtype=torch.long)
for k, v in edge_index_dict.items():
    pyg_data[k].edge_index = torch.tensor(v, dtype=torch.long).T
    
pyg_data = ToUndirected(merge=False)(pyg_data)
# %%
# edge_types = [edge_type for edge_type in pyg_data.edge_types if not edge_type[1].startswith('rev_')]
# rev_edge_types = [(dst, f"rev_{rel}", src) for src, rel, dst in edge_types]
# trans = RandomLinkSplit(
#     num_val=0.0,
#     num_test=0.0,
#     disjoint_train_ratio=0.3,
#     edge_types=edge_types,
#     rev_edge_types=rev_edge_types,
#     add_negative_train_samples=False
# )
# pyg_train, _, _ = trans(pyg_data)
# %%
pyg_valid = pyg_data.clone()
pyg_test = pyg_data.clone()
for target_graph, data in zip([pyg_valid, pyg_test], [valid_edge, test_edge]):
    pos_dict = {}
    neg_dict = {}
    for i in tqdm.tqdm(range(len(data['head_type']))):
        dom, ran = data['head_type'][i], data['tail_type'][i]
        h, t = data['head'][i].item(), data['tail'][i].item()
        hn, tn = data['head_neg'][i].tolist(), data['tail_neg'][i].tolist()
        r = rel_map[data['relation'][i].item()]
        key = (dom, r, ran)
        if key in pos_dict:
            pos_dict[key].append((h, t))
            neg_dict[key].append((hn, tn))
        else:
            pos_dict[key] = [(h, t)]
            neg_dict[key] = [(hn, tn)]
    # break
    for k in pos_dict:
        nbr_ex = len(pos_dict[k])
        nbr_neg = len(neg_dict[k][0][0])
        pos_ex = torch.tensor(pos_dict[k], dtype=torch.long).T
        neg_ex = torch.tensor(neg_dict[k], dtype=torch.long).permute([1,0,2])
        # ex = torch.tensor(pos_dict[k] + neg_dict[k], dtype=torch.long).T
        labels = torch.tensor([1.] * nbr_ex + [0.] * nbr_ex * nbr_neg, dtype=torch.float)
        target_graph[k].edge_label_index = torch.cat((pos_ex, neg_ex.reshape((2,-1))), dim=1)
        target_graph[k].edge_label = labels

        target_graph[k].pos_edge_label_index = pos_ex
        target_graph[k].neg_edge_label_index = neg_ex

# %%
with open(os.path.join(BASE, 'dataset/biokg/biokg_pyg.pkl'), 'wb') as fo:
    pickle.dump({'train': pyg_data, 'valid': pyg_valid}, fo)
    # pickle.dump({'train': pyg_train, 'valid': pyg_valid}, fo)
    # pickle.dump({'train': pyg_train, 'full': pyg_data, 'valid': {'edge_label_dict': pyg_valid.edge_label_dict, 'edge_label_index_dict': pyg_valid.edge_label_index_dict}, 'test': {'edge_label_dict': pyg_test.edge_label_dict, 'edge_label_index_dict': pyg_test.edge_label_index_dict}}, fo)
# %%
with open(os.path.join(BASE, 'dataset/biokg/biokg_pyg_test.pkl'), 'wb') as fo:
    pickle.dump(pyg_test, fo)
# %%
