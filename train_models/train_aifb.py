# %%
import torch
import os, pickle, copy
from torch.cuda import is_available
import tqdm, time
from torch.nn import BCEWithLogitsLoss, Sigmoid

from ogb.linkproppred import Evaluator
import numpy as np
import torch.nn.functional as F

from aifb_model import SingleRGCNDistMult, SingleRGCNComplEx, SingleRGCNNN

from loss_functions import box_loss

if is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE)

# %%
with open(os.path.join(BASE, 'dataset/aifb/aifb_graph.pkl'), 'rb') as fi:
    g = pickle.load(fi)
    train_graph = g['train'].contiguous()
    valid_graph = g['valid'].contiguous()
    test_graph = g['test'].contiguous()
    gci0 = g['gci0']
    max_id = g['max_id']

for v in gci0.values():
    v = v.to(device)

# %%

def sample_neg_ex(num_pos, max_h, max_t, neg_ratio=10, device='cpu',
                  head=None, tail=None):
    """
    Negative sampling for KG edges.

    Args:
        num_pos: number of positive edges
        max_h: number of head nodes
        max_t: number of tail nodes
        neg_ratio: negatives per positive
        head: optional tensor of heads to FIX (shape [1, num_neg] or [1, num_pos])
        tail: optional tensor of tails to FIX (same logic)

    Returns:
        Tensor of shape [2, num_neg]
    """

    num_neg = neg_ratio * num_pos

    # Normalize shape → [1, N]
    if head is not None:
        head = head.to(device)
        if head.dim() == 1:
            head = head.unsqueeze(0)

        if head.shape[1] == num_pos:
            head = head.repeat_interleave(neg_ratio, dim=1)

        heads = head
    else:
        heads = torch.randint(0, max_h, (1, num_neg), device=device)

    if tail is not None:
        tail = tail.to(device)
        if tail.dim() == 1:
            tail = tail.unsqueeze(0)

        if tail.shape[1] == num_pos:
            tail = tail.repeat_interleave(neg_ratio, dim=1)

        tails = tail
    else:
        tails = torch.randint(0, max_t, (1, num_neg), device=device)

    return torch.cat([heads, tails], dim=0)

def get_index_splits(idx_len, message_ratio):
    perm = torch.randperm(idx_len)
    # print(perm)
    split_size = int(np.ceil(idx_len * (1-message_ratio)))
    # print(idx_len / split_size)
    # print(idx_len // split_size)
    nbr_splits = int(np.ceil(idx_len / split_size))
    prev = 0
    for _ in range(nbr_splits):
        label_indices = perm[prev:prev+split_size]
        mask = torch.ones(idx_len, dtype=torch.bool)
        mask[label_indices] = False
        message_indices = torch.arange(idx_len)[mask]
        # print(indices, complement)
        prev = prev+split_size

        yield message_indices, label_indices


# %%
metrics = {'train_losses': [], 'val_metrics': [], 'sem_losses': [],
           'box_losses': [], 'neg_sem_losses': []}
LR = 1e-2
REGULARIZATION = 1e-3
SEM_WEIGHT = 1e-1
NEG_WEIGHT = 0#1e-1
VAL_BATCH_SIZE = 500000

pos_weight = torch.tensor([500.], device=device)

# semantic on all but last layer, when complex

print(f"SEM_WEIGHT: {SEM_WEIGHT}")
print(f"NEG_WEIGHT: {NEG_WEIGHT}")
SUBCLASS_LINKS = False

if SEM_WEIGHT == 0 and not SUBCLASS_LINKS:
    for dom in train_graph.node_id_dict:
        train_graph[dom].node_id = train_graph[dom].node_id[:max_id]
        valid_graph[dom].node_id = valid_graph[dom].node_id[:max_id]
        test_graph[dom].node_id = test_graph[dom].node_id[:max_id]

print(f"subclass links: {SUBCLASS_LINKS}")

if SUBCLASS_LINKS:
    print("using subclass links in graphs, setting SEM_WEIGHT to 0.0")
    SEM_WEIGHT = 0.0
    for k, v in gci0.items():
        sc = v.T.to('cpu')
        train_graph[k, 'subClassOf', k].edge_index = sc
        train_graph[k, 'rev_subClassOf', k].edge_index = sc.flip(0)
        valid_graph[k, 'subClassOf', k].edge_index = sc
        valid_graph[k, 'rev_subClassOf', k].edge_index = sc.flip(0)
        test_graph[k, 'subClassOf', k].edge_index = sc
        test_graph[k, 'rev_subClassOf', k].edge_index = sc.flip(0)

        gci0[k] = v.to('cpu')

train_graph = train_graph.to(device)
valid_graph = valid_graph.to(device)

message_edge_types=[e for e, v in train_graph.edge_index_dict.items() if v.shape[1] > 1]
gnn_channels = 64
nn_channels = [64, 32, 8]
node_dim = 32
DEPTH = 3
EPOCHS = 1000
neg_ratio = 200
print(f"gnn channels: {gnn_channels}")
print(f"nn channels: {nn_channels}")
print(f"node dim: {node_dim}")
print(f"val batch size: {VAL_BATCH_SIZE}")
predicted_edge_types = [('c', 'publication', 'c')]

for k, v in train_graph.node_id_dict.items():
    train_graph[k].node_id = torch.arange(len(v))
    valid_graph[k].node_id = torch.arange(len(v))
    test_graph[k].node_id = torch.arange(len(v))
edge_types = list(set(e[1] for e in train_graph.edge_types))
NUM_BASES = 40
print(f"num bases: {NUM_BASES}")
# decoder = 'distmult'
decoder = 'complex'
# decoder = 'NN'
if decoder == 'distmult':
    model = SingleRGCNDistMult(
        num_nodes=train_graph['c'].node_id.max().item()+1,
        embedding_dim=node_dim,
        hidden_dim=gnn_channels,
        edge_types=edge_types,
        num_layers=DEPTH,
        num_bases=NUM_BASES,
        dropout=0,
    )
elif decoder == 'complex':
    model = SingleRGCNComplEx(
        num_nodes=train_graph['c'].node_id.max().item()+1,
        embedding_dim=node_dim,
        hidden_dim=gnn_channels,
        edge_types=edge_types,
        num_layers=DEPTH,
        num_bases=NUM_BASES,
        dropout=0,
    )
elif decoder == 'NN':
    model = SingleRGCNNN(
        num_nodes=train_graph['c'].node_id.max().item()+1,
        embedding_dim=node_dim,
        hidden_dim=gnn_channels,
        edge_types=edge_types,
        num_layers=DEPTH,
        num_bases=NUM_BASES,
        dropout=0,
        nn_layers=nn_channels
    )

loss_func = BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)


model.to(device)
sigmoid = Sigmoid()
print(f"neg ratio: {neg_ratio}")
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=REGULARIZATION)

scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=40, threshold=0.002,
    cooldown=10, min_lr=1e-5)


ev = Evaluator(name='ogbl-biokg')

with torch.no_grad():
    edges_to_predict = train_graph.edge_index_dict[('c', 'publication', 'c')]
    _ = model(train_graph, edges_to_predict, return_embs=False)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
nbr_params = count_trainable_params(model)
print("Trainable params:", nbr_params)
best_mrr = -np.inf
# %%
MESSAGE_RATIO = 0.85
NEG_VAL_RATIO = int(valid_graph[predicted_edge_types[0]].neg_edge_label_index.shape[1] // valid_graph[predicted_edge_types[0]].pos_edge_label_index.shape[1])
print(f"neg val ratio: {NEG_VAL_RATIO}")

nbr_pred_edges = train_graph[predicted_edge_types[0]].edge_index.shape[1]
split_train_graph = copy.copy(train_graph).to(device)
since_improved = 0
for epoch in tqdm.tqdm(range(EPOCHS)):
    epoch_loss = 0
    epoch_sem_loss = 0
    total_sem_loss = total_neg_sem_loss = 0
    total_epoch_loss = total_epoch_examples = 0
    total_sem_ex = total_neg_sem_ex = 0

    epoch_layer_losses = []
    for message_edges, label_edges in get_index_splits(nbr_pred_edges, MESSAGE_RATIO):
        pos_edges = train_graph[predicted_edge_types[0]].edge_index[:, label_edges]
        rev = (predicted_edge_types[0][2], 'rev_'+predicted_edge_types[0][1], predicted_edge_types[0][0])
        message_edge_index = train_graph[predicted_edge_types[0]].edge_index[:, message_edges]
        rev_message_edge_index = train_graph[rev].edge_index[:, message_edges]
        neg_edges = sample_neg_ex(
                len(label_edges),
                max_id,
                max_id,
                # head=pos_edges[0,:],
                neg_ratio=neg_ratio,
                device=v.device
            )

        neg_edges = torch.cat((neg_edges, sample_neg_ex(
                len(label_edges),
                max_id,
                max_id,
                head=pos_edges[0,:],
                neg_ratio=neg_ratio,
                device=v.device
            )), dim=-1)
        neg_edges = torch.cat((neg_edges, sample_neg_ex(
                len(label_edges),
                max_id,
                max_id,
                tail=pos_edges[1,:],
                neg_ratio=neg_ratio,
                device=v.device
            )), dim=-1)
        split_train_graph[predicted_edge_types[0]].edge_index = message_edge_index
        split_train_graph[rev].edge_index = rev_message_edge_index
        edge_label_indices = torch.cat((pos_edges, neg_edges), dim=1)
        edge_labels = torch.cat((torch.ones_like(pos_edges[0,:], dtype=torch.float), torch.zeros_like(neg_edges[0,:], dtype=torch.float)))
        split_train_graph[predicted_edge_types[0]].edge_label_index = edge_label_indices
        split_train_graph[predicted_edge_types[0]].edge_label = edge_labels

        optimizer.zero_grad()
        edges_to_predict = split_train_graph.edge_label_index_dict[('c', 'publication', 'c')]
        if SEM_WEIGHT > 0:
            y, x_dicts = model(split_train_graph, edges_to_predict, return_embs=True)
            if decoder != 'NN':
                x_dicts = x_dicts[:-1]
            sem_loss, neg_sem_loss, layer_losses, nbr_sem = box_loss(x_dicts,
                                gci0, loss_type='distance',
                                neg=(NEG_WEIGHT > 0), return_layer_loss=True)
            epoch_layer_losses.append(layer_losses)
            total_sem_loss += sem_loss.detach().item()
            if NEG_WEIGHT > 0:
                total_neg_sem_loss += neg_sem_loss.detach().item()
                total_neg_sem_ex += nbr_sem[1]
            total_sem_ex += nbr_sem[0]
        else:
            
            y = model(split_train_graph, edges_to_predict, return_embs=False)

        all_labels = torch.tensor([], device=device)
        for k, v in split_train_graph.edge_label_dict.items():
            all_labels = torch.cat((all_labels, v.to(device)), dim=0)

        loss = loss_func(y.squeeze(), all_labels)
        preds = sigmoid(y.detach()).cpu().numpy().round()
        epoch_loss += loss.detach().item()
        if SEM_WEIGHT > 0:
            sl = SEM_WEIGHT * (sem_loss + NEG_WEIGHT * neg_sem_loss)
            loss += sl
            epoch_sem_loss += sl.detach().item()

        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.detach().item()
        total_epoch_examples += len(all_labels)

        if device != 'cpu':
            if SEM_WEIGHT > 0:
                del x_dicts, all_labels, y
            else:
                del all_labels, y
            torch.cuda.empty_cache()
    print()
    print(f"epoch train loss: {epoch_loss}")
    print(f"epoch sem loss: {epoch_sem_loss}")

    with torch.no_grad():
        val_data = {'y_pred_pos': torch.tensor([], device=device), 'y_pred_neg': torch.tensor([], device=device)}
        for k in valid_graph.pos_edge_label_index_dict.keys():
            pos = valid_graph[k].pos_edge_label_index
            pos_pred = model(valid_graph, pos, return_embs=False)
            val_data['y_pred_pos'] = torch.cat((val_data['y_pred_pos'], pos_pred), dim=0)
            neg = valid_graph[k].neg_edge_label_index
            val_batches = int(np.ceil(neg.shape[1]/VAL_BATCH_SIZE))
            
            idx = 0
            
            for i in range(val_batches):
                neg_data = neg[:,idx:idx+VAL_BATCH_SIZE]
                neg_pred = model(valid_graph, neg_data, return_embs=False)
                val_data['y_pred_neg'] = torch.cat((val_data['y_pred_neg'],
                                            neg_pred.reshape((-1,NEG_VAL_RATIO))), dim=0)
                idx += VAL_BATCH_SIZE
        val_data['y_pred_pos'] = val_data['y_pred_pos'].squeeze()
        val = ev.eval(val_data)
        print(f"MRR: {val['mrr_list'].mean()}")
        print(f"hits@1: {val['hits@1_list'].mean()}")
        print(f"hits@3: {val['hits@3_list'].mean()}")
        print(f"hits@10: {val['hits@10_list'].mean()}", flush=True)

        metrics['val_metrics'].append(val['mrr_list'].mean().item())
        metrics['train_losses'].append(total_epoch_loss / total_epoch_examples)
        if SEM_WEIGHT > 0:
            metrics['box_losses'].append(epoch_layer_losses)
            metrics['sem_losses'].append(total_sem_loss / total_sem_ex)
            if NEG_WEIGHT > 0:
                metrics['neg_sem_losses'].append(total_neg_sem_loss / total_neg_sem_ex)
            else:
                metrics['neg_sem_losses'].append(0)

        if epoch > 100:
            scheduler.step(val['mrr_list'].mean())
        if device != 'cpu':
            del val_data, pos, neg
            torch.cuda.empty_cache()

        if val['mrr_list'].mean() > best_mrr:
            best_mrr = val['mrr_list'].mean()
            print("copying model...")
            best_model = copy.deepcopy(model)
            since_improved = 0
        else:
            since_improved += 1
            if since_improved > 50:
                print("Early stopping due to no improvement")
                break

print(f"Best MRR: {best_mrr}")
    # break
# %%
if SEM_WEIGHT > 0:
    boxes = metrics['box_losses']
    keys = boxes[0][0][0]['pos'].keys()
    if NEG_WEIGHT > 0:
        all_boxes = [{pn: {k: [b[-1][l][pn][k] for b in boxes] for k in keys} for pn in ('pos', 'neg')} for l in range(len(boxes[0][0]))]
    else:
        all_boxes = [{'pos': {k: [b[-1][l]['pos'][k] for b in boxes] for k in keys}} for l in range(len(boxes[0][0]))]
    metrics['box_losses'] = all_boxes
# %%
best_model.to(device)
test_graph = test_graph.to(device)

NEG_TEST_RATIO = int(test_graph[predicted_edge_types[0]].neg_edge_label_index.shape[1] // test_graph[predicted_edge_types[0]].pos_edge_label_index.shape[1])
print(f"neg test ratio: {NEG_TEST_RATIO}")

with torch.no_grad():
    test_data = {'y_pred_pos': torch.tensor([], device=device), 'y_pred_neg': torch.tensor([], device=device)}
    for k in test_graph.pos_edge_label_index_dict.keys():
        pos = test_graph[k].pos_edge_label_index
        pos_pred = best_model(test_graph, pos, return_embs=False)
        test_data['y_pred_pos'] = torch.cat((test_data['y_pred_pos'], pos_pred), dim=0)
        neg = test_graph[k].neg_edge_label_index
        val_batches = int(np.ceil(neg.shape[1]/VAL_BATCH_SIZE))
        idx = 0
        for i in range(val_batches):
            neg_data = neg[:,idx:idx+VAL_BATCH_SIZE]#,:].reshape((2, -1))
            neg_pred = best_model(test_graph, neg_data, return_embs=False)
            test_data['y_pred_neg'] = torch.cat((test_data['y_pred_neg'],
                                        neg_pred.reshape((-1,NEG_TEST_RATIO))), dim=0)
            idx += VAL_BATCH_SIZE

    test_data['y_pred_pos'] = test_data['y_pred_pos'].squeeze()


    test = ev.eval(test_data)
    print("Test set results:")
    print(f"MRR: {test['mrr_list'].mean()}")
    print(f"hits@1: {test['hits@1_list'].mean()}")
    print(f"hits@3: {test['hits@3_list'].mean()}")
    print(f"hits@10: {test['hits@10_list'].mean()}", flush=True)
# %%
print(f"Num bases: {NUM_BASES}")
print(f"node dim: {node_dim}")
print(f"gnn channels: {gnn_channels}")
print(f"sem weight: {SEM_WEIGHT}")
print(f"neg weight: {NEG_WEIGHT}")
print(f"pos weight: {pos_weight.item()}")
print(f"neg ratio: {neg_ratio}")
print(f"regularization: {REGULARIZATION}")
print(f"nn channels: {nn_channels}")
print(f"decoder: {decoder}")
print(f"subclass links: {SUBCLASS_LINKS}")
print(f"depth: {DEPTH}")
print("Trainable params:", count_trainable_params(best_model))

# %%
metrics['test_metrics'] = test['mrr_list'].mean().item()
metrics['trainable_params'] = nbr_params
file_name = time.strftime("%Y%m%d-%H%M%S")
fn = os.path.join(BASE, f'results/metrics-aifb/metrics_{time.strftime("%Y%m%d-%H%M%S")}.pkl')
print(f"saving metrics to {fn}...")
with open(fn, 'wb') as fi:
    pickle.dump(metrics, fi)