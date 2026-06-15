# %%
import torch
import json, os, pickle
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
BASE = os.path.dirname(os.path.abspath(__file__))
print(BASE)
NEG_VAL_RATIO = 1000

# %%
def load_triples(path):
    triples = []
    with open(path, "r") as f:
        for line in f:
            h, r, t = map(int, line.strip().split())
            triples.append([h, r, t])
    return torch.tensor(triples, dtype=torch.long)

train_triples = load_triples("train_triples.txt").to(device)
with open('val_data.json', 'r') as fi:
    val_data = json.load(fi)
val_data['pos'] = torch.tensor(val_data['pos']).to(device)
val_data['neg'] = torch.tensor(val_data['neg']).to(device)

SUBCLASS_LINKS = False
if SUBCLASS_LINKS:

    with open(os.path.join(BASE, 'aifb_graph.pkl'), 'rb') as fi:
        gci0 = pickle.load(fi)['gci0']

    sub_idx = train_triples[:,1].max().item() + 1
    for i, v in enumerate(gci0.values()):
        r_i = (sub_idx + i) * torch.ones((len(v),1), dtype=torch.long).to(device)
        triples = torch.cat([v[:,:1], r_i, v[:,1:]], dim=-1)
        print(triples.shape)
        train_triples = torch.cat((train_triples, triples), dim=0)
    
# %%
if device == 'cuda':
    LR = 1e-3
    EPOCHS = 200
    VAL_BATCH_SIZE = 2**11
    TRAIN_BATCH_SIZE = 2**11
    DIMS = 1024
else:
    DIMS = 256
    LR = 1e-2
    EPOCHS = 500
    VAL_BATCH_SIZE = 2**12
    TRAIN_BATCH_SIZE = 2**15
# DIMS = 2
n_entities = max((train_triples.max(),val_data['pos'].max(), val_data['neg'].max())).item() + 1
n_relations = len(train_triples[:,1].unique())

print('n_entities', n_entities)
print('n_relations', n_relations)
print('DIMS', DIMS)
print(f"subclass links: {SUBCLASS_LINKS}")
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

since_improved = 0
for epoch in tqdm.tqdm(range(EPOCHS)):
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
                                tail_index=neg_data[:,2]).reshape((-1,NEG_VAL_RATIO))
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
                            tail_index=neg_data[:,2]).reshape((-1,NEG_VAL_RATIO))
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

# %%
print(f"dims: {DIMS}")
print(f"Trainable params: {count_trainable_params(model)}")
print(f"subclass links: {SUBCLASS_LINKS}")