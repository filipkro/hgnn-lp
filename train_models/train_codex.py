# %%
import torch
import json, os, pickle, time
from torch.cuda import is_available
import tqdm
from torch.nn import BCEWithLogitsLoss, Sigmoid
from ogb.linkproppred import Evaluator
import numpy as np
from codex_model import Classifier, MultiDomainComplEx, MultiDomainRotatE

from torch_geometric.transforms import ToUndirected
import copy
import torch.nn.functional as F

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
CLASS_MAX = 77951
node_max_ids = {}
node_max_ids['c'] = 77951

# %%
with open(os.path.join(BASE, 'dataset/codex/graphs.pkl'), 'rb') as fi:
    g = pickle.load(fi)
    train_graph = g['train'].contiguous()#.to(device)
    valid_graph = g['val'].contiguous()#.to(device)
    test_graph = g['test'].contiguous()
    gci0 = g['gci0']
    print(g.keys())
# %%
train_graph = ToUndirected(merge=False)(train_graph)
valid_graph = ToUndirected(merge=False)(valid_graph)
test_graph = ToUndirected(merge=False)(test_graph)
del test_graph[('c','rev_P3095','c')].edge_label_index
# %%
for k,v in gci0.items():
    gci0[k] = v.to(device)


with open(os.path.join(BASE, 'dataset/codex/val_negs.json'), 'r') as fi:
    negs = json.load(fi)
    for k, v in negs.items():
        edge_type = tuple(k.split('_'))
        valid_graph[edge_type].neg_edge_label_index = torch.tensor(v)#, device=device)

NEG_VAL_RATIO = valid_graph[edge_type].neg_edge_label_index.shape[1] // valid_graph[edge_type].edge_label_index.shape[1]

with open(os.path.join(BASE, 'dataset/codex/test_negs.json'), 'r') as fi:
    negs = json.load(fi)
    for k, v in negs.items():
        edge_type = tuple(k.split('_'))
        test_graph[edge_type].neg_edge_label_index = torch.tensor(v)
# %%
metrics = {'train_losses': [], 'val_metrics': [], 'sem_losses': [],
           'box_losses': [], 'neg_sem_losses': []}

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

def get_index_splits(idx_lens, message_ratio):
    nbr_splits = int(np.ceil(1 / (1-message_ratio)))
    # prev = 0
    prev = {k: 0 for k in idx_lens.keys()}
    perm = {k: torch.randperm(idx_len) for k, idx_len in idx_lens.items()}
    for _ in range(nbr_splits):
        indices = {}
        for k, idx_len in idx_lens.items():
            split_size = int(np.ceil(idx_len * (1-message_ratio)))
            label_indices = perm[k][prev[k]:prev[k]+split_size]
            prev[k] = prev[k]+split_size
            mask = torch.ones(idx_len, dtype=torch.bool)
            mask[label_indices] = False
            message_indices = torch.arange(idx_len)[mask]
            indices[k] = (message_indices, label_indices)
        # print(indices, complement)
        # prev = prev+split_size
        
        yield indices
        # yield message_indices, label_indices

# %%
MODEL = 'complex' #'classifier', 'rotate'
#MODEL = 'rotate' #'classifier', 'rotate'
MODEL = 'classifier'
# MODEL = 'rgcn_complex'
#MODEL = 'rgcn_nn'
EPOCHS = 300
LR = 1e-1
REGULARIZATION = 1e-3
SEM_WEIGHT = 0#1e-1
NEG_WEIGHT = 1e-1
VAL_BATCH_SIZE = 500000
DISJOINT_RATIO = 0.995
# NEG_VAL_RATIO = 500
neg_ratio = 150
num_batches = 100 
pos_weight = torch.tensor([200.], device=device)
MESSAGE_RATIO = DISJOINT_RATIO
SUBCLASS_LINKS = False


print(f"SEM_WEIGHT: {SEM_WEIGHT}")
print(f"NEG_WEIGHT: {NEG_WEIGHT}")

if device == 'cpu':
    gnn_channels = [2]
    nn_channels = [2]
    node_dim = 2
    HIDDEN_DIM = 2
    EPOCHS = 2
    neg_ratio = 2
    num_batches = 2
    num_bases = 8
    DISJOINT_RATIO = 0.5
else:
    HIDDEN_DIM = 64
    gnn_channels = [64, 128]
    nn_channels = [32, 8]
    node_dim = 32
    num_bases = 30
    
print(f"gnn channels: {gnn_channels}")
print(f"nn channels: {nn_channels}")
print(f"node dim: {node_dim}")
print(f"hidden dim: {HIDDEN_DIM}")
print(f"val batch size: {VAL_BATCH_SIZE}")
print(f"scoring head: {MODEL}")
print(f"disjoint or message ratio: {DISJOINT_RATIO}")
print(f"num bases: {num_bases}")
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
    
    
predicted_edge_types = list(set(valid_graph.edge_label_index_dict.keys()).union(set(test_graph.edge_label_index_dict.keys())))
loss_func = BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
if MODEL == 'classifier':
    model = Classifier(
        gnn_channels=gnn_channels,
        nn_channels=nn_channels,
        embedding_dims={k: node_dim for k in train_graph.node_types},
        number_classes={k: len(v) for k,v in train_graph.node_id_dict.items()},
        message_edge_types=train_graph.edge_types,
        predicted_edge_types=predicted_edge_types,
        custom=False,
        device=device
    )
elif MODEL == 'complex':
    model = MultiDomainComplEx(
        gnn_channels=gnn_channels,
        embedding_dims={k: node_dim for k in train_graph.node_types},
        number_classes={k: len(v) for k,v in train_graph.node_id_dict.items()},
        message_edge_types=train_graph.edge_types,
        predicted_edge_types=predicted_edge_types,
        hidden_dim=HIDDEN_DIM,
        use_layernorm=True,
        device=device
    )
    
elif MODEL == 'rotate':
    model = MultiDomainRotatE(
        gnn_channels=gnn_channels,
        embedding_dims={k: node_dim for k in train_graph.node_types},
        number_classes={k: len(v) for k,v in train_graph.node_id_dict.items()},
        message_edge_types=train_graph.edge_types,
        predicted_edge_types=predicted_edge_types,
        hidden_dim=HIDDEN_DIM,
        gamma=12.0,
        use_layernorm=True,
        device=device
    )
    loss_func = F.logsigmoid
else:
    raise NotImplementedError(f"not implemented model {MODEL}")

model.to(device)
sigmoid = Sigmoid()
print(f"pos weight: {pos_weight}")
print(f"neg ratio: {neg_ratio}")
print(f"num batches: {num_batches}")
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=REGULARIZATION)

ev = Evaluator(name='ogbl-biokg')

k = predicted_edge_types[0]
#init_edges = {k: train_graph.edge_index_dict[k][:,100].to(device)}
init_edges = {k: train_graph.edge_index_dict[k].to(device)}
train_graph = train_graph.to(device)
with torch.no_grad():
    _ = model(train_graph, edges_to_predict=init_edges, return_embs=False)
train_graph = train_graph.to('cpu')

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
nbr_params = count_trainable_params(model)
print("Trainable params:", nbr_params)

best_mrr = -np.inf

# %%
edge_idx_lens = {k: train_graph.edge_index_dict[k].shape[1] for k in predicted_edge_types}
split_train_graph = copy.copy(train_graph).to(device)
since_improved = 0
for epoch in tqdm.tqdm(range(EPOCHS)):
    valid_graph = valid_graph.to('cpu')
    epoch_loss = 0
    epoch_sem_loss = 0
    total_sem_loss = total_neg_sem_loss = 0
    total_epoch_loss = total_epoch_examples = 0
    total_sem_ex = total_neg_sem_ex = 0

    if epoch == 5:
        LR = 5e-2
        for g in optimizer.param_groups:
            g['lr'] = LR
        print(f"updated learning rate: {LR}")
    elif epoch == 40:
        LR = 1e-2
        for g in optimizer.param_groups:
            g['lr'] = LR
        print(f"updated learning rate: {LR}")
    elif epoch == int(EPOCHS/4):
        LR = LR / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR
        print(f"updated learning rate: {LR}")
    elif epoch == int(EPOCHS/2):
        LR = LR / 5
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR
        print(f"updated learning rate: {LR}")

    epoch_layer_losses = []
    for edge_indices in get_index_splits(edge_idx_lens, MESSAGE_RATIO):
        # message_edges, label_edges
        for k, (message_edges, label_edges) in edge_indices.items():
            pos_edges = train_graph[k].edge_index[:, label_edges].to(device)
            message_edge_index = train_graph[k].edge_index[:, message_edges].to(device)
            rev = (k[2], 'rev_' + k[1], k[0])
            rev_message_edge_index = train_graph[rev].edge_index[:, message_edges].to(device)
            head_max_id = node_max_ids[k[0]]
            tail_max_id = node_max_ids[k[2]]
            neg_edges = sample_neg_ex(
                    len(label_edges),
                    head_max_id,
                    tail_max_id,
                    # head=pos_edges[0,:],
                    neg_ratio=neg_ratio,
                    device=device
                )

            split_train_graph[k].edge_index = message_edge_index
            split_train_graph[rev].edge_index = rev_message_edge_index
            edge_label_indices = torch.cat((pos_edges, neg_edges), dim=1)
            edge_labels = torch.cat((torch.ones_like(pos_edges[0,:], dtype=torch.float, device=device), torch.zeros_like(neg_edges[0,:], dtype=torch.float, device=device)))
            split_train_graph[k].edge_label_index = edge_label_indices
            split_train_graph[k].edge_label = edge_labels
        optimizer.zero_grad()
        
        if SEM_WEIGHT > 0:
            y, x_dicts = model(split_train_graph, edges_to_predict=split_train_graph.edge_label_index_dict, return_embs=True)
        
        else:    
            y = model(split_train_graph, edges_to_predict=split_train_graph.edge_label_index_dict, return_embs=False)

        all_labels = torch.tensor([], device=device)
        for k, v in split_train_graph.edge_label_dict.items():
            all_labels = torch.cat((all_labels, v.to(device)), dim=0)
        if MODEL == 'rotate':
            neg_mask = all_labels == 0
            pos_mask = all_labels == 1
            scores = y.squeeze()
            loss = -(pos_weight * loss_func(scores[pos_mask])).sum() - loss_func(-scores[neg_mask]).sum()
        else:
            loss = loss_func(y.squeeze(), all_labels)
        epoch_loss += loss.detach().item()
        if SEM_WEIGHT > 0:
            sem_loss, neg_sem_loss, layer_losses, nbr_sem = box_loss(x_dicts,
                                gci0, loss_type='distance',
                                neg=(NEG_WEIGHT > 0), return_layer_loss=True)
            epoch_layer_losses.append(layer_losses)
            total_sem_loss += sem_loss.detach().item()
            total_neg_sem_loss += neg_sem_loss.detach().item()
            total_sem_ex += nbr_sem[0]
            total_neg_sem_ex += nbr_sem[1]
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
    print(f"epoch train loss: {epoch_loss}")
    print(f"epoch sem loss: {epoch_sem_loss}")

    valid_graph = valid_graph.to(device)
    with torch.no_grad():
        val_data = {'y_pred_pos': torch.tensor([], device=device), 'y_pred_neg': torch.tensor([], device=device)}
        for k in valid_graph.edge_label_index_dict.keys():
            pos = valid_graph[k].edge_label_index
            pos_pred = model(valid_graph, edges_to_predict={k: pos}, return_embs=False)
            val_data['y_pred_pos'] = torch.cat((val_data['y_pred_pos'], pos_pred), dim=0)
            neg = valid_graph[k].neg_edge_label_index
            val_batches = int(np.ceil(neg.shape[1]/VAL_BATCH_SIZE))
            idx = 0
            for i in range(val_batches):
                neg_data = neg[:,idx:idx+VAL_BATCH_SIZE]#,:].reshape((2, -1))
                neg_pred = model(valid_graph, edges_to_predict={k: neg_data}, return_embs=False)
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
            metrics['neg_sem_losses'].append(total_neg_sem_loss / total_neg_sem_ex)

        if device != 'cpu':
            del val_data, pos, neg, neg_data
            torch.cuda.empty_cache()

        if val['mrr_list'].mean() > best_mrr:
            best_mrr = val['mrr_list'].mean()
            print('copying model...', flush=True)
            best_model = copy.deepcopy(model)
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= 20:
                print('Early stopping...')
                break

print(f"Best MRR: {best_mrr}")

if SEM_WEIGHT > 0:
    boxes = metrics['box_losses']
    keys = boxes[0][0][0]['pos'].keys()
    all_boxes = [{pn: {k: [b[-1][l][pn][k] for b in boxes] for k in keys} for pn in ('pos', 'neg')} for l in range(len(boxes[0][0]))]
    metrics['box_losses'] = all_boxes

# %%
if device != 'cpu':
    del train_graph, valid_graph, model
    torch.cuda.empty_cache()
best_model.to(device)
test_graph = test_graph.to(device)

VAL_BATCH_SIZE = int(VAL_BATCH_SIZE / 2)

with torch.no_grad():
    test_data = {'y_pred_pos': torch.tensor([], device=device), 'y_pred_neg': torch.tensor([], device=device)}
    for k in test_graph.edge_label_index_dict.keys():
        pos = test_graph[k].edge_label_index
        
        pos_pred = best_model(test_graph, edges_to_predict={k: pos}, return_embs=False)
        test_data['y_pred_pos'] = torch.cat((test_data['y_pred_pos'], pos_pred), dim=0)
        neg = test_graph[k].neg_edge_label_index
        val_batches = int(np.ceil(neg.shape[1]/VAL_BATCH_SIZE))
        idx = 0
        for i in range(val_batches):
            neg_data = neg[:,idx:idx+VAL_BATCH_SIZE]#,:].reshape((2, -1))
            neg_pred = best_model(test_graph, edges_to_predict={k: neg_data}, return_embs=False)
            test_data['y_pred_neg'] = torch.cat((test_data['y_pred_neg'],
                                        neg_pred.reshape((-1,NEG_VAL_RATIO))), dim=0)
            idx += VAL_BATCH_SIZE
    test_data['y_pred_pos'] = test_data['y_pred_pos'].squeeze()

    test = ev.eval(test_data)
    print("Test set results:")
    print(f"MRR: {test['mrr_list'].mean()}")
    print(f"hits@1: {test['hits@1_list'].mean()}")
    print(f"hits@3: {test['hits@3_list'].mean()}")
    print(f"hits@10: {test['hits@10_list'].mean()}", flush=True)
# %%
metrics['test_metrics'] = test['mrr_list'].mean().item()
metrics['trainable_params'] = nbr_params
metrics['subclass_links'] = SUBCLASS_LINKS
print(f"subclass links: {SUBCLASS_LINKS}")
file_name = time.strftime("%Y%m%d-%H%M%S")
fn = os.path.join(BASE, f'results/metrics-codex/metrics_{time.strftime("%Y%m%d-%H%M%S")}.pkl')
print(f"saving metrics to {fn}...")
with open(fn, 'wb') as fi:
    pickle.dump(metrics, fi)
