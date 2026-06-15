# %%
import pandas as pd
import os
import tqdm
import rdflib
from rdflib.namespace import RDFS
import json
import csv
import requests
from bs4 import BeautifulSoup
import time
import itertools
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)
# %%
fp = os.path.join(BASE, 'dataset/ogbl_biokg/mapping/drug_entidx2name.csv')
print(fp)
# %%
drug_map = pd.read_csv(fp, sep=',', index_col='ent idx').to_dict()['ent name']
# %%
CIDS = set([int(k.replace('CID', '')) for k in drug_map.values()])
cid2smile = {}

with open("CID-SMILES") as f:
    reader = csv.reader(f, delimiter="\t")

    for row in tqdm.tqdm(reader):
        if int(row[0]) in CIDS:
            cid2smile[int(row[0])] = row[1]
        # print(row)
        # break
print(len(cid2smile))
print(len(CIDS))

# %%
BASE = "http://classyfire.wishartlab.com"

fc = 0
c = 0

cid2smile_split = {fc: {}}
for k,v in cid2smile.items():
    if c == 100:
        f.close()
        c = 0
        fc += 1
        cid2smile_split[fc] = {}
    cid2smile_split[fc][k] = v
    c += 1

print(len(cid2smile_split))
# %%
full_results = {}
spinner = itertools.cycle(['|', '/', '-', '\\'])
query_ids = {}
cool_downs = 0
session = requests.Session()
for it, smiles_list in cid2smile_split.items():
    if it in full_results:
        continue
    time.sleep(1)

    # -------------------------------------------------
    # 1. Get CSRF token
    # -------------------------------------------------
    r = session.get(BASE)
    while not r.ok:
        print(f'waiting for cool down ({cool_downs})... {next(spinner)}', end='\r', flush=True)
        time.sleep(5 + cool_downs*0.1)
        cool_downs += 1
        r = session.get(BASE)
    soup = BeautifulSoup(r.text, "html.parser")
    token = soup.find("meta", {"name": "csrf-token"})["content"]

    # -------------------------------------------------
    # 2. Build multi-SMILES query input
    # -------------------------------------------------
    # smiles_list = {
    #     "a": "CC(=O)OC(CC([O-])=O)C[N+](C)(C)C",
    #     "b": "C1CC(=O)NC1C(=O)O",
    #     "c": "C1C(C(OC1N2C=CC(=O)NC2=O)CO)O"
    # }

    query_input = "\n".join(
        f"{label}\t{smiles}"
        for label, smiles in smiles_list.items()
    )

    # -------------------------------------------------
    # 3. Submit query
    # -------------------------------------------------
    payload = {
        "utf8": "✓",
        "authenticity_token": token,
        "query[query_type]": "STRUCTURE",
        "query[query_input]": query_input,
        "query[label]": "",
        "commit": "Submit"
    }
    time.sleep(1)
    r = session.post(
        f"{BASE}/queries",
        data=payload,
        allow_redirects=False
    )

    while not r.ok:
        print(f'waiting for cool down ({cool_downs})... {next(spinner)}', end='\r', flush=True)
        time.sleep(5 + cool_downs*0.1)
        cool_downs += 1
        r = session.post(
            f"{BASE}/queries",
            data=payload,
            allow_redirects=False
        )

    query_id = r.headers["Location"].split("/")[-1]
    print(f"It: {it}, Query ID: {query_id}")

    query_ids[it] = query_id

print("Querying for results...")
time.sleep(5)
for it, query_id in query_ids.items():
    if it in full_results:
        continue
    # -------------------------------------------------
    # 4. Poll for results
    # -------------------------------------------------
    result_url = f"{BASE}/queries/{query_id}.json"
    
    while True:
        r = session.get(result_url)
        
        print(f'{r.status_code} - processing... {next(spinner)}', end='\r', flush=True)
        time.sleep(5)
        if r.status_code == 200:
            print(results['classification_status'], end='\r')
            results = r.json()
            if results['classification_status'] == "Done":
                break

    print()
    # print(results)
    full_results[it] = results
    print(f"Query {it} done")
print(len(full_results))
# %%
for it, query_id in query_ids.items():
    if it in full_results:
        continue
    # -------------------------------------------------
    # 4. Poll for results
    # -------------------------------------------------
    result_url = f"{BASE}/queries/{query_id}.json"
    
    while True:
        r = session.get(result_url)
        
        print(f'processing... {next(spinner)}', end='\r', flush=True)
        time.sleep(1)
        if r.status_code == 200:
            results = r.json()
            if results['classification_status'] == "Done" or True:
                break

    print()
    # print(results)
    full_results[it] = results
    print(f"Query {it} done")
print(len(full_results))
# %%
session = requests.Session()
for it, query_id in query_ids.items():
    if it in full_results:# or it in [64, 87]:
        continue
    # -------------------------------------------------
    # 4. Poll for results
    # -------------------------------------------------
    result_url = f"{BASE}/queries/{query_id}.json"
    
    while True:
        time.sleep(5)
        r = session.get(result_url)
        
        print(f'{r.status_code} - processing... {next(spinner)}', end='\r', flush=True)
        if r.status_code == 200:
            results = r.json()
            if results['classification_status'] == "Done" or True:
                print()
                print(results['classification_status'])
                break

    # print(results)
    full_results[it] = results
    print(f"Query {it} done")
print(len(full_results))
# %%

with open('datafiles/classyfire_query_ids.json', 'w') as fo:
    json.dump(query_ids, fo)
# %%
with open('datafiles/classyfire_full_results.json', 'w') as fo:
    json.dump(full_results, fo)
# %%
with open('datafiles/cid2smile_split.json', 'w') as fo:
    json.dump(cid2smile_split, fo)
# %%
missing_ids = [64, 87]
small_batches = {}
for k in missing_ids:
    c = 0
    fc = 0
    small_batches[k] = {fc: {}}
    for i in cid2smile_split[k].values():
        if c == 10:
            c = 0
            fc += 1
            small_batches[k][fc] = {}
        small_batches[k][fc][c] = i
        c += 1

# %%
spinner = itertools.cycle(['|', '/', '-', '\\'])
# query_ids = {}
cool_downs = 0
session = requests.Session()
batch_results = {}
batch_query_ids = {}
for iid, smile_dict in small_batches.items():
    batch_query_ids[iid] = {}
    for it, smiles_list in smile_dict.items():
        if it in batch_query_ids[iid]:
            continue
        time.sleep(0.5)

        # -------------------------------------------------
        # 1. Get CSRF token
        # -------------------------------------------------
        r = session.get(BASE)
        while not r.ok:
            print(f'waiting for cool down ({cool_downs})... {next(spinner)}', end='\r', flush=True)
            time.sleep(0.5 + cool_downs*0.1)
            cool_downs += 1
            r = session.get(BASE)
        soup = BeautifulSoup(r.text, "html.parser")
        token = soup.find("meta", {"name": "csrf-token"})["content"]


        query_input = "\n".join(
            f"{label}\t{smiles}"
            for label, smiles in smiles_list.items()
        )

        # -------------------------------------------------
        # 3. Submit query
        # -------------------------------------------------
        payload = {
            "utf8": "✓",
            "authenticity_token": token,
            "query[query_type]": "STRUCTURE",
            "query[query_input]": query_input,
            "query[label]": "",
            "commit": "Submit"
        }
        time.sleep(0.5)
        r = session.post(
            f"{BASE}/queries",
            data=payload,
            allow_redirects=False
        )

        while not r.ok:
            print(f'waiting for cool down ({cool_downs})... {next(spinner)}', end='\r', flush=True)
            time.sleep(1 + cool_downs*0.1)
            cool_downs += 1
            r = session.post(
                f"{BASE}/queries",
                data=payload,
                allow_redirects=False
            )

        query_id = r.headers["Location"].split("/")[-1]
        print(f"It: {it}, Query ID: {query_id}")

        batch_query_ids[iid][it] = query_id

print("Querying for results...")
time.sleep(3)
for iid, batch_q_ids in batch_query_ids.items():
    batch_results[iid] = {}
    for it, query_id in batch_query_ids.items():
        if it in batch_results[iid]:
            continue
        # -------------------------------------------------
        # 4. Poll for results
        # -------------------------------------------------
        result_url = f"{BASE}/queries/{query_id}.json"
        
        while True:
            r = session.get(result_url)
            
            print(f'{r.status_code} - processing... {next(spinner)}', end='\r', flush=True)
            time.sleep(1)
            if r.status_code == 200:
                print(results['classification_status'], end='\r')
                results = r.json()
                if results['classification_status'] == "Done":
                    batch_results[iid][it] = results
                    break
            elif r.status_code == 500:
                print()
                print(f"Code 500, iid: {iid}, it: {it}")

        print()
        # print(results)
        print(f"Query {it} done")
print(len(batch_results))
print([len(v) for v in batch_results.values()])
# %%

# schema:
# {cid: {kingdom, superclass, class, subclass, , [intermediate_nodes], direct_parent}}

cid_info = {}
not_found = {}
keys = ['kingdom', 'superclass', 'class', 'subclass', 'direct_parent']
for ki, q_res in full_results.items():
    assert q_res['number_of_elements'] == len(q_res['entities']), ki

    for c in q_res['entities']:
        if 'identifier' not in c:
            if ki in not_found:
                not_found[ki].append(c)
            else:
                not_found[ki] = [c]
        else:
            cid = c['identifier']

            cid_info[cid] = {'intermediate_nodes': c['intermediate_nodes']}
            for k in keys:
                cid_info[cid][k] = c[k]['chemont_id'] if c[k] != None else None
        

print(len(cid_info))
print(len(not_found))
print(sum([len(v) for v in not_found.values()]))
# %%
csv_files = ['12765032.csv', '12765033.csv']
print(len(cid_info))
for f in csv_files:
    with open(f, 'r') as fi:
        reader = csv.reader(fi, delimiter=",")
        for row in tqdm.tqdm(reader):
            if row[0] == 'CompoundID':
                continue
            cid = row[0]
            if cid not in cid_info:
                cid_info[cid] = {'intermediate_nodes': []}
            
            if 'Kingdom' in row[2]:
                cid_info[cid]['kingdom'] = row[1]
            elif 'Superclass' in row[2]:
                cid_info[cid]['superclass'] = row[1]
            elif 'Class' in row[2]:
                cid_info[cid]['class'] = row[1]
            elif 'Subclass' in row[2]:
                cid_info[cid]['subclass'] = row[1]
            elif 'Direct_parent' in row[2]:
                cid_info[cid]['direct_parent'] = row[1]
            elif 'Intermediate_nodes' in row[2]:
                cid_info[cid]['intermediate_nodes'].append({'name': row[2].split('Intermediate_nodes: ')[-1], 'chemont_id': row[1]})
            

            # print(row)
            # assert cid not in cid_info, (cid, f) 
        # break
    # break
print(len(cid_info))
# %%
direct_parents = {k: v['direct_parent'] for k, v in cid_info.items()}
# %%
chem_ont = rdflib.Graph()
chem_ont.parse('datafiles/chemont.ttl')
# %%
prefix = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX obo: <http://purl.obolibrary.org/obo/>

SELECT DISTINCT ?ancestor
WHERE {
    """
classes_in_hierachy = set()
for c in tqdm.tqdm(set(direct_parents.values())):
    if c not in classes_in_hierachy:
        c = c.replace(':', '_')
        classes_in_hierachy.add(c)
        query = prefix + f"obo:{c} rdfs:subClassOf+ ?ancestor .}}"
        qres = chem_ont.query(query)
        for row in qres:
            if isinstance(row[0], rdflib.term.BNode):
                continue
            par = row[0].toPython().split('/')[-1]
            classes_in_hierachy.add(par)

print(len(set(direct_parents.values())))
print(len(classes_in_hierachy))
# %%
cid_hierarchy = rdflib.Graph()
OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')

CID_PREFIX = rdflib.Namespace('http://cid.project-genesis.io#')

for c in classes_in_hierachy:
    s = OBO[c]
    for r in chem_ont.predicate_objects(s):
        if not isinstance(r[1], rdflib.term.BNode):
            cid_hierarchy.add((s, r[0], r[1]))
print(len(cid_hierarchy))
for c,p in direct_parents.items():
    c = CID_PREFIX[c]
    p = OBO[p.replace(':', '_')]
    
    cid_hierarchy.add((c, RDFS.subClassOf, p))
print(len(cid_hierarchy))
# %%
cid_hierarchy.serialize('datafiles/cid_hierarchy.ttl')
# %%
# The following builds the drug class hierarchy encoded in cid_hierarchy.ttl
cid_hierarchy = rdflib.Graph()
cid_hierarchy.parse('datafiles/cid_hierarchy.ttl')
CID_PREFIX = rdflib.Namespace('http://cid.project-genesis.io#')
# %%
# Build index2class and rev_index for all drugs
index2class = {}
for k in drug_map.keys():
    # The CID_PREFIX should match the IRI for these drug classes.
    index2class[k] = CID_PREFIX[f"{k}"].toPython()
rev_index = {v: k for k, v in index2class.items()}

cp = []
curr_key = len(index2class)
for res in tqdm.tqdm(cid_hierarchy.subject_objects(RDFS.subClassOf)):
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

print(f"Checking {len(cp)} pairs...")
for pair in tqdm.tqdm(cp):
    assert (rdflib.URIRef(index2class[pair[0]]), RDFS.subClassOf, rdflib.URIRef(index2class[pair[1]])) in cid_hierarchy, f"Pair {pair} is not in cid_hierarchy"

print("All pairs are in cid_hierarchy.")


# Save cp in a JSON file as instructed
with open('biokg_gci0/gci0_drug.json', 'w') as f:
    json.dump(cp, f)

with open('biokg_gci0/class_index_drug.json', 'w') as f:
    json.dump(index2class, f)

# %%
