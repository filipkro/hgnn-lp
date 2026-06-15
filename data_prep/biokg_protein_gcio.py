# %%

import json
import rdflib
from rdflib.namespace import RDFS
import pandas as pd
import os
import tqdm
# import torch
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
mapping_dir = os.path.join(BASE, 'dataset/ogbl_biokg/mapping')
print(mapping_dir)

prot_dict = pd.read_csv(os.path.join(mapping_dir, 'protein_entidx2name.csv'), sep=',', index_col='ent idx').to_dict()['ent name']

# %%
df = pd.read_csv(os.path.join(BASE, 'datafiles/hgnc_complete_set.txt'), sep='\t')
print(df.head())
# %%
entrez = df['entrez_id']
entrez_set = set(entrez.unique().astype(int))
# %%
c = 0
for k,v in tqdm.tqdm(prot_dict.items()):
    c += int(int(v.split('|')[-1]) not in entrez_set)
print(len(prot_dict))
print(c)
# %%
gene_group = {}
gg_missing = set()
for k,v in tqdm.tqdm(prot_dict.items()):
    gg = []
    for g in v.split('|'):
        if int(g) in entrez_set and not \
                df.loc[df['entrez_id'] == int(g), 'gene_group_id'].isna().item():
            gg.extend(df.loc[df['entrez_id'] == int(g), 'gene_group_id'].item().split('|'))
        else:
            gg_missing.add(v)
    if len(gg) > 0:
        gene_group[k] = gg
print(len(gene_group))
print(len(gg_missing))

# %%

closed_h = pd.read_csv(os.path.join(BASE, 'datafiles/hierarchy_closure.csv'), sep=',')
print(closed_h.head())
# %%
all_groups = set(int(vv) for v in gene_group.values() for vv in v)
# %%
classes_in_h = set()
for row in closed_h.itertuples():
    if row.child_fam_id in all_groups:
        classes_in_h.add(row.parent_fam_id)
print(len(classes_in_h))
print(len(all_groups))
print(len(all_groups.intersection(classes_in_h)))
# %%
open_h = pd.read_csv(os.path.join(BASE, 'datafiles/hierarchy.csv'), sep=',')
print(open_h.head())
# %%

GENE_PREFIX = rdflib.Namespace('http://hgnc.project-genesis.io#')
gene_hierarchy = rdflib.Graph()

a = b = 0
for row in open_h.itertuples():
    if row.child_fam_id in classes_in_h:
        a += 1
        # print(row)
        gene_hierarchy.add((GENE_PREFIX[f'FAMILY_{row.child_fam_id}'], RDFS.subClassOf, GENE_PREFIX[f'FAMILY_{row.parent_fam_id}']))
    else:
        b += 1
print(len(gene_hierarchy))

family = pd.read_csv(os.path.join(BASE, 'datafiles/family.csv'), sep=',')
print(family.head())

for row in family.itertuples():
    # break
    if row.id in classes_in_h:
        if isinstance(row.name, str):
            gene_hierarchy.add((GENE_PREFIX[f"FAMILY_{row.id}"], RDFS.label, rdflib.Literal(row.name)))
        if isinstance(row.desc_comment, str):
            gene_hierarchy.add((GENE_PREFIX[f"FAMILY_{row.id}"], RDFS.comment, rdflib.Literal(row.desc_comment)))


for k, v in gene_group.items():
    for vg in v:
        gene_hierarchy.add((GENE_PREFIX[f"PROT_{k}"], RDFS.subClassOf, GENE_PREFIX[f"FAMILY_{vg}"]))
        

gene_hierarchy.serialize('gene_hierarchy.ttl')


# %%
gene_hierarchy = rdflib.Graph()
gene_hierarchy.parse(os.path.join(BASE, 'datafiles/gene_hierarchy.ttl'), format='turtle')
GENE_PREFIX = rdflib.Namespace('http://hgnc.project-genesis.io#')
# %%
index2class = {}
for k in prot_dict.keys():
    index2class[k] = GENE_PREFIX[f"PROT_{k}"].toPython()
rev_index = {v: k for k,v in index2class.items()}

cp = []
curr_key = len(index2class)
for res in tqdm.tqdm(gene_hierarchy.subject_objects(RDFS.subClassOf)):
    if isinstance(res[0], rdflib.URIRef) and isinstance(res[1], rdflib.URIRef):
        r0 = res[0].toPython()
        r1 = res[1].toPython()
        if r0 in rev_index:
            i0 = rev_index[r0]
        else:
            i0 = curr_key
            index2class[i0] = r0
            rev_index[r0] = i0
            curr_key += 1
        if r1 in rev_index:
            i1 = rev_index[r1]
        else:
            i1 = curr_key
            index2class[i1] = r1
            rev_index[r1] = i1
            curr_key += 1
        cp.append((i0, i1))
    # break
# %%
# Test that rev_index is the inverse of index2class
for k, v in index2class.items():
    assert rev_index[v] == k, f"rev_index is not the inverse of index2class for key {k}, value {v}"

for v, k in rev_index.items():
    assert index2class[k] == v, f"index2class is not the inverse of rev_index for value {v}, key {k}"

print("rev_index and index2class are exact inverses of each other.")

# %%
# cp_tensor = torch.tensor(cp, dtype=torch.long)
# %
# Save cp in a JSON file as instructed
with open('biokg_gci0/gci0_protein.json', 'w') as f:
    json.dump(cp, f)

with open('biokg_gci0/class_index_protein.json', 'w') as f:
    json.dump(index2class, f)

# %%
