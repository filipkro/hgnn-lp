# %%
import pandas as pd
import pickle
import os
import copy

# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
# %%
codex = pd.read_csv(os.path.join(BASE, 'datafiles/codex-l/full.txt'), sep='\t', index_col=False, names=['s','p','o'])
print(codex.head())
print(len(codex))
# %%
wiki_31 = pd.read_csv(os.path.join(BASE, 'datafiles/wikidata5m/full_P31.txt'), sep='\t', index_col=False, names=['s','p','o'])
print(wiki_31.head())
print(len(wiki_31))
# %%
wiki_279 = pd.read_csv(os.path.join(BASE, 'datafiles/wikidata5m/full_P279.txt'), sep='\t', index_col=False, names=['s','p','o'])
print(wiki_279.head())
print(len(wiki_279))
# %%
codex_classes = set(codex['s'].unique()).union(codex['o'].unique())
print(len(codex_classes))
# %%
combined_hierarchy = pd.concat((wiki_279[['s','o']], wiki_31[['s', 'o']]))


# %%
from collections import defaultdict, deque

# Step 1: build adjacency list (child -> [parents])
adj = defaultdict(list)
for s, o in zip(combined_hierarchy['s'].values,
                combined_hierarchy['o'].values):
    adj[s].append(o)

# Step 2: BFS/DFS traversal
hierarchy = set()
seen = set(codex_classes)
queue = deque(codex_classes)
used_classes = copy.deepcopy(codex_classes)

while queue:
    c = queue.popleft()

    for parent in adj.get(c, []):
        edge = (c, parent)
        used_classes.add(parent)
        if edge not in hierarchy:
            hierarchy.add(edge)

        if parent not in seen:
            seen.add(parent)
            queue.append(parent)

print(len(hierarchy))
print(len(used_classes))
# %%
missing = []
for c in codex_classes:
    if c not in adj:
        missing.append(c)

print(len(missing))
# %%
from torch_geometric.data import HeteroData
# %%
index_dict = {k: i for i, k in enumerate(codex_classes)}
start_i = len(index_dict)
for i, c in enumerate(used_classes - codex_classes):
    index_dict[c] = start_i + i
print(len(index_dict))
print(len(used_classes))
# %%
rev = {v:k for k,v in index_dict.items()}
# %%
codex_train = pd.read_csv(os.path.join(BASE, 'datafiles/codex-l/train.txt'), sep='\t',index_col=False, names=['s','p','o'])
# %%
edges = {}
for row in codex_train.itertuples():
    if row[2] in edges:
        edges[row[2]].append((index_dict[row[1]], index_dict[row[3]]))
    else:
        edges[row[2]] = [(index_dict[row[1]], index_dict[row[3]])]

# %%
import torch

# %%
hierarchy_index = [(index_dict[h[0]], index_dict[h[1]]) for h in hierarchy]
gci0 = {'c':torch.tensor(hierarchy_index, dtype=torch.long)}
# %%
graph = HeteroData()
graph['c'].node_id = torch.tensor(range(len(index_dict)), dtype=torch.long)
for r, e in edges.items():
    graph['c', r, 'c'].edge_index = torch.tensor(e).T
    # print(torch.tensor(e).T)
    # break
# %%
val_graph = copy.deepcopy(graph)
test_graph = copy.deepcopy(graph)


# %%
codex_val = pd.read_csv(os.path.join(BASE, 'datafiles/codex-l/valid.txt'), sep='\t',index_col=False, names=['s','p','o'])
# %%
edges_val = {}
for row in codex_val.itertuples():
    if row[2] in edges_val:
        edges_val[row[2]].append((index_dict[row[1]], index_dict[row[3]]))
    else:
        edges_val[row[2]] = [(index_dict[row[1]], index_dict[row[3]])]
for r, e in edges_val.items():
    val_graph['c', r, 'c'].edge_label_index = torch.tensor(e).T
# %%
codex_test = pd.read_csv(os.path.join(BASE, 'datafiles/codex-l/test.txt'), sep='\t',index_col=False, names=['s','p','o'])
# %%
edges_test = {}
for row in codex_test.itertuples():
    if row[2] in edges_test:
        edges_test[row[2]].append((index_dict[row[1]], index_dict[row[3]]))
    else:
        edges_test[row[2]] = [(index_dict[row[1]], index_dict[row[3]])]
for r, e in edges_test.items():
    test_graph['c', r, 'c'].edge_label_index = torch.tensor(e).T
# %%
print(len(gci0['c']))
gci = gci0['c'][gci0['c'][:,0] != index_dict['Q2145290']]
terms2skip = ['Q16889133', 'Q5127848', 'Q19478619', 'Q217594']
for t in terms2skip:
    gci = gci[gci[:,0] != index_dict[t]]
    gci = gci[gci[:,1] != index_dict[t]]
print(len(gci))
gci0['c'] = gci
# %%
with open(os.path.join(BASE, 'dataset/codex/graphs.pkl'), 'wb') as fo:
    pickle.dump({'train': graph, 'val': val_graph, 'test': test_graph, 'gci0': gci0}, fo)
