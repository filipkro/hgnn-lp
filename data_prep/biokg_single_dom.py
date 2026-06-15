# %%
import os, pickle, json
import tqdm
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
# %%
with open(os.path.join(BASE, 'dataset/biokg/biokg_pyg.pkl'), 'rb') as fi:
    d = pickle.load(fi)
    train = d['train']
    val = d['valid']

# %%
with open(os.path.join(BASE, 'dataset/biokg/biokg_pyg_test.pkl'), 'rb') as fi:
    test = pickle.load(fi)
# %%
rmap = {e[1]: i for i,e in enumerate(train.edge_types) if 'rev_' not in e[1]}
# %%
nmap = {k: {} for k in train.node_id_dict}
# %%
nmap = {}
curr = 0
for k,v in train.node_id_dict.items():
    nmap[k] = {nid.item(): nid.item() + curr for nid in v}
    curr = v.max().item() + curr + 1
    # print(k)
    # print(v)
    # break
# %%
triples = []
for e, d in train.edge_index_dict.items():
    # rel = rmap[e[1]]
    triples.extend([(nmap[e[0]][ei[0].item()], rmap[e[1]], nmap[e[2]][ei[1].item()]) for ei in d.T if 'rev_' not in e[1]])
    # print(e[0], e[1], e[2])
    # print(d)
    # break
# %%
# val_pos_triples = []
val_data = {'pos': [], 'neg': []}
for e, d in val.pos_edge_label_index_dict.items():
    # rel = rmap[e[1]]
    if 'rev_' not in e[1]:
        val_data['pos'].extend([(nmap[e[0]][ei[0].item()], rmap[e[1]], nmap[e[2]][ei[1].item()]) for ei in d.T])

        neg_d = val.neg_edge_label_index_dict[e].permute(1, 2, 0)
        # print(neg_d.shape)
        val_data['neg'].extend([[(nmap[e[0]][ei[0].item()], rmap[e[1]], nmap[e[2]][ei[1].item()]) for ei in neg] for neg in neg_d])
        # break

# val_neg_triples = []
# for e, d in val.neg_edge_label_index_dict.items():
#     # rel = rmap[e[1]]
#     val_neg_triples.extend([(nmap[e[0]][ei[0].item()], rmap[e[1]], nmap[e[2]][ei[1].item()]) for ei in d.T if 'rev_' not in e[1]])
# %%
test_data = {'pos': [], 'neg': []}
for e, d in tqdm.tqdm(test.pos_edge_label_index_dict.items()):
    # rel = rmap[e[1]]
    if 'rev_' not in e[1]:
        test_data['pos'].extend([(nmap[e[0]][ei[0].item()], rmap[e[1]], nmap[e[2]][ei[1].item()]) for ei in d.T])

        neg_d = test.neg_edge_label_index_dict[e].permute(1, 2, 0)
        # print(neg_d.shape)
        test_data['neg'].extend([[(nmap[e[0]][ei[0].item()], rmap[e[1]], nmap[e[2]][ei[1].item()]) for ei in neg] for neg in neg_d])
# %%
with open(os.path.join(BASE, 'dataset/biokg/train_triples.txt'), 'w') as fo:
    for t in triples:
        fo.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")
# %%
with open(os.path.join(BASE, 'dataset/biokg/val_data.json'), 'w') as fo:
    json.dump(val_data, fo)
# %%
with open(os.path.join(BASE, 'dataset/biokg/test_data.json'), 'w') as fo:
    json.dump(test_data, fo)
# %%
with open(os.path.join(BASE, 'dataset/biokg/node_id_map.json'), 'w') as fo:
    json.dump(nmap, fo)
# %%
