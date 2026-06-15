# %%
import pickle, json, os
import tqdm
import torch
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
# %%
with open(os.path.join(BASE, 'dataset/aifb/aifb_graph.pkl'), 'rb') as f:
    g = pickle.load(f)
    train_val = g['valid'].contiguous()
    test_graph = g['test'].contiguous()

# %%
edge_types = set([e[1] for e in test_graph.edge_types if 'rev_' not in e[1]])
edge_map = {e: i for i, e in enumerate(edge_types)}
# %%
train = []
for k,v in tqdm.tqdm(train_val.edge_index_dict.items()):
    if 'rev_' not in k[1]:
        for p in v.T.tolist():
            train.append((p[0], edge_map[k[1]], p[1]))

# %%
val = []
val = {'pos': [], 'neg': []}
NEG_SIZE = 1000
for k,v in tqdm.tqdm(train_val.pos_edge_label_index_dict.items()):
    if 'rev_' in k[1]:
        continue
    prev_idx = 0
    negs = train_val.neg_edge_label_index_dict[k].T
    r_i = edge_map[k[1]] * torch.ones((len(negs),1), dtype=torch.long)
    negs = torch.cat([negs[:,:1], r_i, negs[:,1:]], dim=-1)

    for p in v.T.tolist():
        val['pos'].append((p[0], edge_map[k[1]], p[1]))
        nn = negs[prev_idx:prev_idx+NEG_SIZE]
        prev_idx += NEG_SIZE
        val['neg'].append(nn.tolist())


print(len(val))
test = []
test = {'pos': [], 'neg': []}
for k,v in tqdm.tqdm(test_graph.pos_edge_label_index_dict.items()):
    if 'rev_' in k[1]:
        continue
    prev_idx = 0

    negs = test_graph.neg_edge_label_index_dict[k].T
    r_i = edge_map[k[1]] * torch.ones((len(negs),1), dtype=torch.long)
    negs = torch.cat([negs[:,:1], r_i, negs[:,1:]], dim=-1)
    for p in v.T.tolist():
        test['pos'].append((p[0], edge_map[k[1]], p[1]))
        nn = negs[prev_idx:prev_idx+NEG_SIZE]
        prev_idx += NEG_SIZE
        test['neg'].append(nn.tolist())
print(len(test))

# %%
with open(os.path.join(BASE, 'dataset/aifb/train_triples.txt'), 'w') as f:
    for t in train:
        f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")

with open(os.path.join(BASE, 'dataset/aifb/val_data.json'), 'w') as f:
    json.dump(val, f)

with open(os.path.join(BASE, 'dataset/aifb/test_data.json'), 'w') as f:
    json.dump(test, f)
