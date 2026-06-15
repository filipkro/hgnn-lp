# %%
import pandas as pd
import json
import rdflib
import tqdm
import os, csv
from rdflib.namespace import RDFS, OWL
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
# %%
fp = os.path.join(BASE, 'dataset/ogbl_biokg/mapping/disease_entidx2name.csv')
# fp = os.path.join(BASE, 'disease_entidx2name.csv')
print(fp)
# %%
disease_map = pd.read_csv(fp, sep=',', index_col='ent idx').to_dict()['ent name']
# %%
OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')
OBOINOWL = rdflib.Namespace('http://www.geneontology.org/formats/oboInOwl#')

# %%

parents = {}
children = {}
broader = {}
is_a = {}
inv_is_a = {}

with open("datafiles/MRREL.RRF") as f:
    reader = csv.reader(f, delimiter="|")

    for row in tqdm.tqdm(reader):
        if row[3] == "PAR":
            child = row[0]
            parent = row[4]
            if parent != child:
                if child in parents:
                    parents[child].add(parent)
                else:
                    parents[child] = {parent}
        elif row[3] == "CHD":
            parent = row[0]
            child = row[4]
            if parent != child:
                if parent in children:
                    children[parent].add(child)
                else:
                    children[parent] = {child}
        elif row[7] == 'isa':
            child = row[0]
            parent = row[4]
            if parent != child:
                if child in is_a:
                    is_a[child].add(parent)
                else:
                    is_a[child] = {parent}
        elif row[7] == 'inverse_isa':
            parent = row[0]
            child = row[4]
            if parent != child:
                if parent in inv_is_a:
                    inv_is_a[parent].add(child)
                else:
                    inv_is_a[parent] = {child}
        elif row[3] == "RB":
            child = row[0]
            parent = row[4]
            if parent != child:
                if child in broader:
                    broader[child].add(parent)
                else:
                    broader[child] = {parent}

# %%
all_children = set(parents.keys())
all_parents = set(children.keys())

all_inv = set(inv_is_a.keys())
all_isa = set(is_a.keys())

print(len(all_children - all_parents))
print(len(all_parents - all_children))
top_level = all_parents - all_children
classes_in_hierarchy = all_children.union(all_parents)


# %%
def add_hierarchy_w_b(c, classes):
    if c not in classes and c in parents:#  not in top_level:
        classes.add(c)
        # if c not in top_level:
        for p in parents[c]:
            if p not in classes:
                # print(p)
                classes = add_hierarchy_w_b(p, classes)
    elif c not in classes and c in broader:
        classes.add(c)
        # if c not in top_level:
        for p in broader[c]:
            if p not in classes:
                classes = add_hierarchy_w_b(p, classes)
    elif c not in classes and c in is_a:
        classes.add(c)
        # if c not in top_level:
        for p in is_a[c]:
            if p not in classes:
                classes = add_hierarchy_w_b(p, classes)
    return classes

# %%
bset = set(broader.keys())
print(len(bset - classes_in_hierarchy))
print(len(bset))
all_broader = set()
for v in broader.values():
    all_broader.update(v)
print(len(all_broader))

# %%
# classes_w_parents = set(parents.keys()).union(set(broader.keys())).union(set(is_a.keys()))
classes_in_h_b = classes_in_hierarchy.union(all_broader).union(bset)
print(len(classes_in_hierarchy))
print(len(classes_in_h_b))
classes_w_parents = classes_in_h_b.union(set(is_a.keys()))
print(len(classes_w_parents))
classes_hb = set()
missing_both = []
for d in tqdm.tqdm(disease_map.values()):
    if d in classes_w_parents:
        classes_hb = add_hierarchy_w_b(d, classes_hb)
    else:
        missing_both.append(d)

print(len(classes_hb))
# print(len(classes))
print(len(missing_both))

# %%
PREFIX = rdflib.Namespace('http://umls.project-genesis.io#')
RDFS = rdflib.Namespace('http://www.w3.org/2000/01/rdf-schema#')
umls_hierarchy = rdflib.Graph()
umls_hierarchy.bind('umls', PREFIX)
# missing2 = []
for c in classes_hb:
    if c in parents:
        it = parents[c]
    elif c in broader:
        it = broader[c]
    elif c in is_a:
        it = is_a[c]
    else:
        raise ValueError(c)
    for p in it:
        umls_hierarchy.add((PREFIX[c], RDFS.subClassOf, PREFIX[p]))

# %%
umls_hierarchy.serialize('datafiles/disease_hierarchy.ttl')
# %%
# The following is adapted from the "gene_hierarchy" code, replacing all gene_hierarchy with umls_hierarchy.
umls_hierarchy = rdflib.Graph()
umls_hierarchy.parse('datafiles/disease_hierarchy.ttl')

# %%
PREFIX = rdflib.Namespace('http://umls.project-genesis.io#')
index2class = {}
for k, v in disease_map.items():
    # The PREFIX should match the IRI for these disease classes.
    index2class[k] = PREFIX[v].toPython()
rev_index = {v: k for k, v in index2class.items()}

cp = []
curr_key = len(index2class)
for res in tqdm.tqdm(umls_hierarchy.subject_objects(RDFS.subClassOf)):
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

# Test that rev_index is the inverse of index2class
for k, v in index2class.items():
    assert rev_index[v] == k, f"rev_index is not the inverse of index2class for key {k}, value {v}"

for v, k in rev_index.items():
    assert index2class[k] == v, f"index2class is not the inverse of rev_index for value {v}, key {k}"

print("rev_index and index2class are exact inverses of each other.")

# Save cp in a JSON file as instructed
with open('biokg_gci0/gci0_disease.json', 'w') as f:
    json.dump(cp, f)

with open('biokg_gci0/class_index_disease.json', 'w') as f:
    json.dump(index2class, f)


# %%
