# %%
import os, pickle, json
import tqdm
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

num_ents = 0
# dom_map = {k:}
dom_map = {}
for dom in train.node_id_dict:
    ents = 0
    dom_map[dom] = num_ents
    for e in train.edge_types:
        curr = 0
        if e[0] == dom and 'rev_' not in e[1]:
            curr = max((train.edge_index_dict[e][0].max().item(), val.edge_index_dict[e][0].max().item(),test.edge_index_dict[e][0].max().item())) + 1
            # print(curr)
        elif e[2] == dom and 'rev_' not in e[1]:
            curr = max((train.edge_index_dict[e][1].max().item(), val.edge_index_dict[e][1].max().item(),test.edge_index_dict[e][1].max().item())) + 1
            # print(curr)
        # print(curr, ents)
        ents = max(ents, curr)
    print(dom, ents)
    num_ents += ents
print(num_ents)
print(dom_map)
# %%
clean_triples = []
for e, d in train.edge_index_dict.items():
    # rel = rmap[e[1]]
    clean_triples.extend([(dom_map[e[0]] + ei[0].item(), rmap[e[1]], dom_map[e[2]] + ei[1].item()) for ei in d.T if 'rev_' not in e[1]])
# %%
val_data = {'pos': [], 'neg': []}
for e, d in val.pos_edge_label_index_dict.items():
    # rel = rmap[e[1]]
    if 'rev_' not in e[1]:
        val_data['pos'].extend([(dom_map[e[0]] + ei[0].item(), rmap[e[1]], dom_map[e[2]] + ei[1].item()) for ei in d.T])

        neg_d = val.neg_edge_label_index_dict[e].permute(1, 2, 0)
        # print(neg_d.shape)
        val_data['neg'].extend([[(dom_map[e[0]] + ei[0].item(), rmap[e[1]], dom_map[e[2]] + ei[1].item()) for ei in neg] for neg in neg_d])

# %%
test_data = {'pos': [], 'neg': []}
for e, d in tqdm.tqdm(test.pos_edge_label_index_dict.items()):
    # rel = rmap[e[1]]
    if 'rev_' not in e[1]:
        test_data['pos'].extend([(dom_map[e[0]] + ei[0].item(), rmap[e[1]], dom_map[e[2]] + ei[1].item()) for ei in d.T])

        neg_d = test.neg_edge_label_index_dict[e].permute(1, 2, 0)
        # print(neg_d.shape)
        test_data['neg'].extend([[(dom_map[e[0]] + ei[0].item(), rmap[e[1]], dom_map[e[2]] + ei[1].item()) for ei in neg] for neg in neg_d])

# %%
with open(os.path.join(BASE, 'dataset/biokg/train_triples_fix.txt'), 'w') as fo:
    for t in clean_triples:
        fo.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")
# %%
with open(os.path.join(BASE, 'dataset/biokg/val_data_fix.json'), 'w') as fo:
    json.dump(val_data, fo)
# %%
with open(os.path.join(BASE, 'dataset/biokg/test_data_fix.json'), 'w') as fo:
    json.dump(test_data, fo)
