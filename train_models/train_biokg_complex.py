# %%
import torch
import json, os
from torch.cuda import is_available
import tqdm
import copy

from ogb.linkproppred import Evaluator
import numpy as np

from torch_geometric.nn.kge import ComplEx

if is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)

# %%

def load_triples(path):
    triples = []
    with open(path, "r") as f:
        for line in f:
            h, r, t = map(int, line.strip().split())
            triples.append([h, r, t])
    return torch.tensor(triples, dtype=torch.long)


# %%
SUBCLASS_LINKS = False
if SUBCLASS_LINKS:

    gci0 = {}
    gci0_dir = os.path.join(BASE, 'biokg_gci0')
    for fname in os.listdir(gci0_dir):
        if fname.startswith('gci0'):
            with open(os.path.join(gci0_dir, fname), 'r') as f:
                key = fname.split('_')[-1].split('.')[0]
                gci0[key] = torch.tensor(json.load(f), dtype=torch.long,
                                        device=device)

    with open(os.path.join(BASE, 'dataset/biokg/node_id_map.json'), 'r') as fi:
        node_map = json.load(fi)
    
    train_triples = load_triples(os.path.join(BASE, "dataset/biokg/train_triples.txt"))#.to(device)

    with open(os.path.join(BASE, 'dataset/biokg/val_data.json'), 'r') as fi:
        val_data = json.load(fi)
    val_data['pos'] = torch.tensor(val_data['pos']).to(device)
    val_data['neg'] = torch.tensor(val_data['neg']).to(device)


    sub_idx = train_triples[:,1].max().item() + 1
    # for 
    # assert False
    print(train_triples.shape)
    for i, (k, v) in enumerate(gci0.items()):
        r_i = (sub_idx + i) * torch.ones((len(v),1), dtype=torch.long).to(device)
        # print(v[:,:1])
        # print(v[:,:1].apply_(lambda x: node_map[k][str(x)]))
        triples = torch.cat([v[:,:1].apply_(lambda x: node_map[k][str(x)]), r_i, v[:,1:].apply_(lambda x: node_map[k][str(x)])], dim=-1)
        print(triples.shape)
        train_triples = torch.cat((train_triples, triples), dim=0)
    print(train_triples.shape)
else:
    train_triples = load_triples(os.path.join(BASE, "dataset/biokg/train_triples_fix.txt"))#.to(device)

    with open(os.path.join(BASE, 'dataset/biokg/val_data_fix.json'), 'r') as fi:
        val_data = json.load(fi)
    val_data['pos'] = torch.tensor(val_data['pos']).to(device)
    val_data['neg'] = torch.tensor(val_data['neg']).to(device)
train_triples = train_triples.to(device)
# %%
if device == 'cuda':
    LR = 1e-3
    EPOCHS = 200
    VAL_BATCH_SIZE = 2**12
    TRAIN_BATCH_SIZE = 2**12
    DIMS = 512
else:
    DIMS = 2
    LR = 1e-1
    EPOCHS = 10
    VAL_BATCH_SIZE = 2**12
    TRAIN_BATCH_SIZE = 2**15
# DIMS = 2
n_entities = max((train_triples.max(),val_data['pos'].max(), val_data['neg'].max())).item() + 1
n_relations = len(train_triples[:,1].unique())

print('n_entities', n_entities)
print('n_relations', n_relations)
print(f"SUBCLASS_LINKS: {SUBCLASS_LINKS}")
print('DIMS', DIMS)
# %%
model = ComplEx(num_nodes=n_entities, num_relations=n_relations,
                hidden_channels=DIMS).to(device)


num_val_batches = int(np.ceil(len(val_data['pos']) / VAL_BATCH_SIZE))
num_train_batches = int(np.ceil(len(train_triples) / TRAIN_BATCH_SIZE))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable params:", count_trainable_params(model))
ev = Evaluator(name='ogbl-biokg')

# %%
best_mrr = -np.inf
# fix batching for training 
since_improved = 0
for epoch in tqdm.tqdm(range(EPOCHS)):
    # break
    # maybe batch edge_label_index
    epoch_loss = 0
    epoch_sem_loss = 0
    model.train()
    total_loss = total_examples = 0
    prev_idx = 0
    for batch in range(num_train_batches):
        optimizer.zero_grad()

        batch_triples = train_triples[prev_idx:prev_idx+TRAIN_BATCH_SIZE]
        loss = model.loss(batch_triples[:,0], batch_triples[:,1], batch_triples[:,2])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * batch_triples[:,0].numel()
        total_examples += batch_triples[:,0].numel()  
        prev_idx += TRAIN_BATCH_SIZE    


    # loss = model.loss(train_triples[:,0], train_triples[:,1], train_triples[:,2])
    # loss.backward()
    # optimizer.step()
    print(f"train loss: {total_loss}")
    with torch.no_grad():
        val_scores = {'y_pred_pos': torch.tensor([], device=device), 'y_pred_neg': torch.tensor([], device=device)}
        prev_idx = 0
        for i in range(num_val_batches):
            pos_data = val_data['pos'][prev_idx:prev_idx+VAL_BATCH_SIZE]
            neg_data = val_data['neg'][prev_idx:prev_idx+VAL_BATCH_SIZE].reshape((-1,3))
            pos_preds = model(head_index=pos_data[:,0], rel_type=pos_data[:,1],
                                tail_index=pos_data[:,2])
            neg_preds = model(head_index=neg_data[:,0], rel_type=neg_data[:,1],
                                tail_index=neg_data[:,2]).reshape((-1,500))
            val_scores['y_pred_pos'] = torch.cat((val_scores['y_pred_pos'], pos_preds), dim=0)
            val_scores['y_pred_neg'] = torch.cat((val_scores['y_pred_neg'], neg_preds), dim=0)
            prev_idx += VAL_BATCH_SIZE

        val_scores['y_pred_pos'] = val_scores['y_pred_pos'].squeeze()

        val = ev.eval(val_scores)
        print(f"MRR: {val['mrr_list'].mean()}")
        print(f"hits@1: {val['hits@1_list'].mean()}")
        print(f"hits@3: {val['hits@3_list'].mean()}")
        print(f"hits@10: {val['hits@10_list'].mean()}", flush=True)

        if val['mrr_list'].mean() > best_mrr:
            best_mrr = val['mrr_list'].mean()
            best_model = copy.deepcopy(model)
            since_improved = 0
            print('copying model...')
        else:
            since_improved += 1
            if since_improved >= 20:
                print('Early stopping...')
                break

print(f"Best MRR: {best_mrr}")

# %%
best_model.to(device)
# %%
with open('test_data.json', 'r') as fi:
    test_data = json.load(fi)
test_data['pos'] = torch.tensor(test_data['pos']).to(device)
test_data['neg'] = torch.tensor(test_data['neg']).to(device)
# %%
with torch.no_grad():
    test_scores = {'y_pred_pos': torch.tensor([], device=device), 'y_pred_neg': torch.tensor([], device=device)}
    prev_idx = 0
    for i in range(num_val_batches):
        pos_data = test_data['pos'][prev_idx:prev_idx+VAL_BATCH_SIZE]
        neg_data = test_data['neg'][prev_idx:prev_idx+VAL_BATCH_SIZE].reshape((-1,3))
        pos_preds = best_model(head_index=pos_data[:,0], rel_type=pos_data[:,1],
                            tail_index=pos_data[:,2])
        neg_preds = best_model(head_index=neg_data[:,0], rel_type=neg_data[:,1],
                            tail_index=neg_data[:,2]).reshape((-1,500))
        test_scores['y_pred_pos'] = torch.cat((test_scores['y_pred_pos'], pos_preds), dim=0)
        test_scores['y_pred_neg'] = torch.cat((test_scores['y_pred_neg'], neg_preds), dim=0)
        prev_idx += VAL_BATCH_SIZE

    test_scores['y_pred_pos'] = test_scores['y_pred_pos'].squeeze()

    val = ev.eval(test_scores)
    print("Test set results:")
    print(f"MRR: {val['mrr_list'].mean()}")
    print(f"hits@1: {val['hits@1_list'].mean()}")
    print(f"hits@3: {val['hits@3_list'].mean()}")
    print(f"hits@10: {val['hits@10_list'].mean()}", flush=True)


print(f"SUBCLASS_LINKS: {SUBCLASS_LINKS}")