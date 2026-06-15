# %%
import mowl
mowl.init_jvm("12g")
from mowl.datasets.el import ELDataset
from mowl.datasets.base import PathDataset
import pandas as pd
import rdflib
import tqdm
import json, re
from rdflib.namespace import RDFS, OWL
# import torch
import os, sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../kg-box-emb/code/embeddings')))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../kg-box-emb/code/data_prep')))
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
# %%
mapping_dir = os.path.join(BASE, 'dataset/ogbl_biokg/mapping')
print(mapping_dir)
# %%
function_map = pd.read_csv(os.path.join(mapping_dir, 'function_entidx2name.csv'), sep=',', index_col='ent idx').to_dict()['ent name']

# %%
OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')
OBOINOWL = rdflib.Namespace('http://www.geneontology.org/formats/oboInOwl#')

# %%
g_early = rdflib.Graph()
g_early.parse('datafiles/go_early_2017.ttl')
g_ext = rdflib.Graph()
g_ext.parse('datafiles/go-ext.ttl')
g = rdflib.Graph()
g.parse('datafiles/go.ttl')
g20 = rdflib.Graph()
g20.parse('datafiles/go2020.ttl')
g15 = rdflib.Graph()
g15.parse('datafiles/go2015.ttl')

# %%
graphs = {'2022': g_ext, '2020': g20, '2018': g, '2017': g_early, '2015': g15}
ordering = ['2022', '2020', '2018', '2017', '2015']


def in_graph(go, graph=None, el_dict=None):
    return (rdflib.URIRef(go), None, None) in graph and \
            (rdflib.URIRef(go), OWL.deprecated, rdflib.Literal(True)) not in graph

def in_graph_22(go, graph=None, el_dict=None):
    return go in el_dict

check_in_graph = {'2022': in_graph_22, '2020': in_graph, '2018': in_graph,
                  '2017': in_graph, '2015': in_graph}
# %%
kg_fp = os.path.join(BASE, 'datafiles/go-ext.ttl')
data = PathDataset(kg_fp)
el_dataset = ELDataset(data.ontology)
el_dataset.load()
used_classes = {k: set() for k in ordering}
missing = {k: [] for k in ordering}
corrected_dict = {}

for y in ordering:
    print(f"Processing {y}...")
    graph = graphs[y]
    for k, v in function_map.items():
        go = 'http://purl.obolibrary.org/obo/' + v.replace(':', '_')
        if k in corrected_dict:
            continue
        if check_in_graph[y](go, graph=graph,
                             el_dict=el_dataset.class_index_dict):
            used_classes[y].add(go)
            corrected_dict[k] = v
        else:
            consider = list(graph.objects(rdflib.URIRef(go), OBO.IAO_0100001))
            if len(consider) > 1:
                print(go)
                print(consider)
                assert False
            elif len(consider) > 0:
                new_go = consider[0].toPython().split('/')[-1].replace('_', ':')
                if (consider[0], OWL.deprecated, rdflib.Literal(True)) in graph:
                    missing[y].append(go)
                else:
                    used_classes[y].add(consider[0].toPython())
                    corrected_dict[k] = new_go
            else:
                missing[y].append(go)
    print(f'Finished {y}. Found {len(corrected_dict)} of {len(function_map)} classes.')
for y in ordering:
    print(f"{y}: {len(used_classes[y])} used, {len(missing[y])} missing, total: {len(used_classes[y]) + len(missing[y])}")
print(f'total: {len(function_map)}')
print(f'corrected: {len(corrected_dict)}')

# %%
classes_in_hierachy = {k: set() for k in ordering}
# classes_in_hierachy_refined = {k: set() for k in ordering}
classes_in_hierachy_refined = {}
prefix = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?ancestor
WHERE {
    """
# classes_in_hierachy = set()
for i, y in enumerate(ordering):
    print(f"Processing {y}...")
    graph = graphs[y]
    for c in tqdm.tqdm(used_classes[y]):
        if c not in classes_in_hierachy[y]:
            classes_in_hierachy[y].add(c)
            query = prefix + f"<{c}> rdfs:subClassOf+ ?ancestor .}}"
            qres = graph.query(query)
            for row in qres:
                if isinstance(row[0], rdflib.term.BNode):
                    continue
                classes_in_hierachy[y].add(row[0].toPython())
        # break
    refined = classes_in_hierachy[y]
    for ii in range(i):
        refined = refined - classes_in_hierachy[ordering[ii]]
    classes_in_hierachy_refined[y] = refined
    print(f"Finished {y}. Total in hierarchy: {len(classes_in_hierachy[y])}, refined: {len(classes_in_hierachy_refined[y])}")

# %%
go_hierarchy = rdflib.Graph()
for y, hierarchy in classes_in_hierachy_refined.items():
    print(f"Processing {y}...")
    graph = graphs[y]
    # hierarchy = hierarchies[y]
    for c in tqdm.tqdm(hierarchy):
        s = rdflib.URIRef(c)
        for r in graph.predicate_objects(s):
            if not isinstance(r[1], rdflib.term.BNode):
                go_hierarchy.add((s, r[0], r[1]))

# %%
print(len(go_hierarchy))
print(len(g_ext))
# %%
go_hierarchy.serialize('datafiles/go_hierarchy.ttl')
# %%
with open('datafiles/go_map.json', 'w') as fo:
    json.dump(corrected_dict, fo)

# %%
go_hierarchy = rdflib.Graph()
go_hierarchy.parse('datafiles/go_hierarchy.ttl')

with open('datafiles/go_map.json', 'r') as fo:
    corrected_dict = json.load(fo)

# %%
# Build a mapping from value to list of keys to find non-unique mappings
from collections import defaultdict

# Step 1: Find duplicates in corrected_dict values
value_to_keys = defaultdict(list)
for k, v in corrected_dict.items():
    value_to_keys[v].append(int(k))

# Quick sanity: inspect the worst-case duplicate count
max_dups = max((len(keys) for keys in value_to_keys.values()), default=0)
print(f"Max duplicate mappings for a single GO id: {max_dups}")

# Step 2: Build index2class with unique suffix for duplicates
OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')
index2class = {}    # key (int) -> OBO URI string (with suffix if necessary)
uri_to_keys = {}    # OBO URI string -> int (for reverse lookup, unique)

# Also, keep a mapping from canonical_uri (without suffix) to all keys for use in cp construction
canonical_uri_to_all_keys = defaultdict(list)

for v, keys in value_to_keys.items():
    base_uri = OBO[f"{v.replace(':', '_')}"].toPython()
    if len(keys) == 1:
        # Unique mapping
        index2class[keys[0]] = base_uri
        uri_to_keys[base_uri] = keys[0]
        canonical_uri_to_all_keys[base_uri].append(keys[0])
    else:
        # Non-unique mapping: add _0, _1, ...
        for i, k in enumerate(keys):
            suffixed_uri = f"{base_uri}_{i}"
            index2class[k] = suffixed_uri
            uri_to_keys[suffixed_uri] = k
            canonical_uri_to_all_keys[base_uri].append(k)

# Step 3: Build perfect inverse mapping (rev_index)
rev_index = {v: k for k, v in index2class.items()}

# Step 4: Build cp with new keying (handle multiple suffixed instances)
cp = []
curr_key = max(index2class.keys(), default=-1) + 1


_suffix_re = re.compile(r"^(.*)_(\d+)$")

def canonicalize_uri(uri: str) -> str:
    """Return canonical (unsuffixed) URI if it exists in canonical_uri_to_all_keys."""
    if uri in canonical_uri_to_all_keys:
        return uri
    m = _suffix_re.match(uri)
    if not m:
        return uri
    candidate = m.group(1)
    return candidate if candidate in canonical_uri_to_all_keys else uri

for res in tqdm.tqdm(go_hierarchy.subject_objects(RDFS.subClassOf)):
    if isinstance(res[0], rdflib.URIRef) and isinstance(res[1], rdflib.URIRef):
        r0 = res[0].toPython()
        r1 = res[1].toPython()
        # r0 and r1 are canonical URIs (potentially with or without suffix)
        # Find all keys (with potential suffixes) for r0 and r1
        r0_keys = canonical_uri_to_all_keys.get(canonicalize_uri(r0), [])
        r1_keys = canonical_uri_to_all_keys.get(canonicalize_uri(r1), [])
        # If r0 is actually one of the suffixed URIs, include only that
        if r0 in rev_index:
            r0_keys = [rev_index[r0]]
        if r1 in rev_index:
            r1_keys = [rev_index[r1]]
        # If none found, add as new
        if not r0_keys:
            index2class[curr_key] = r0
            rev_index[r0] = curr_key
            r0_keys = [curr_key]
            curr_key += 1
        if not r1_keys:
            index2class[curr_key] = r1
            rev_index[r1] = curr_key
            r1_keys = [curr_key]
            curr_key += 1
        # For every r0 (child) key, connect to *all* r1 (parent) keys
        for i0 in r0_keys:
            for i1 in r1_keys:
                cp.append((i0, i1))
    # break

# Test that rev_index is the inverse of index2class
for k, v in index2class.items():
    assert rev_index[v] == k, f"rev_index is not the inverse of index2class for key {k}, value {v}"

for v, k in rev_index.items():
    assert index2class[k] == v, f"index2class is not the inverse of rev_index for value {v}, key {k}"

print("rev_index and index2class are exact inverses of each other.")

print(f"Checking {len(cp)} pairs...")
for pair in tqdm.tqdm(cp):
    assert (rdflib.URIRef(canonicalize_uri(index2class[pair[0]])), RDFS.subClassOf, rdflib.URIRef(canonicalize_uri(index2class[pair[1]]))) in go_hierarchy, f"Pair {pair} is not in go_hierarchy"

print("All pairs are in go_hierarchy.")
# Save cp in a JSON file as instructed
with open('biokg_gci0/gci0_function.json', 'w') as f:
    json.dump(cp, f)

with open('biokg_gci0/class_index_function.json', 'w') as f:
    json.dump(index2class, f)

