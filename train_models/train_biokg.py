# %%
import torch
import json, os, pickle, time
from torch.cuda import is_available
import tqdm
from torch.nn import BCEWithLogitsLoss, Sigmoid
from ogb.linkproppred import Evaluator
import numpy as np
from biokg_model import Classifier, MultiDomainComplEx, MultiDomainRotatE

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
DISEASE_MAX = 10686 + 1
DRUG_MAX = 10532 + 1
FUNCTION_MAX = 45084 + 1
PROTEIN_MAX = 17498 + 1
SIDEEFFECT_MAX = 9968 + 1
node_max_ids = {}
node_max_ids['disease'] = DISEASE_MAX
node_max_ids['drug'] = DRUG_MAX
node_max_ids['function'] = FUNCTION_MAX
node_max_ids['protein'] = PROTEIN_MAX
node_max_ids['sideeffect'] = SIDEEFFECT_MAX

# %%
gci0 = {}
gci0_dir = os.path.join(BASE, 'biokg_gci0')
for fname in os.listdir(gci0_dir):
    if fname.startswith('gci0'):
        with open(os.path.join(gci0_dir, fname), 'r') as f:
            key = fname.split('_')[-1].split('.')[0]
            gci0[key] = torch.tensor(json.load(f), dtype=torch.long,
                                     device=device)
# %%
with open(os.path.join(BASE, 'dataset/biokg/biokg_pyg.pkl'), 'rb') as fi:
    g = pickle.load(fi)
    train_graph = g['train'].contiguous()#.to(device)
    valid_graph = g['valid'].contiguous()#.to(device)

with open(os.path.join(BASE, 'dataset/biokg/biokg_pyg_test.pkl'), 'rb') as fi:
    test_graph = pickle.load(fi).contiguous()


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

    # return all_batches

def get_index_splits(idx_lens, message_ratio):
    nbr_splits = int(np.ceil(1 / (1-message_ratio)))
    
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

        yield indices


# %%
MODEL = 'complex' #'classifier', 'rotate'
#MODEL = 'rotate' #'classifier', 'rotate'
MODEL = 'classifier'
EPOCHS = 250
LR = 1e-1
REGULARIZATION = 1e-2
SEM_WEIGHT = 1e-2
NEG_WEIGHT = 0#1e-1
VAL_BATCH_SIZE = 500000
neg_ratio = 100
# num_batches = 90 
pos_weight = torch.tensor([250.], device=device)
SUBCLASS_LINKS = False 

GCI0_DOMAINS_TO_TRAIN = []# list(gci0.keys())
GCI0_DOMAINS_TO_TRAIN = ['function']


print(f"GCI0_DOMAINS_TO_TRAIN: {GCI0_DOMAINS_TO_TRAIN}")

print(f"SEM_WEIGHT: {SEM_WEIGHT}")
print(f"NEG_WEIGHT: {NEG_WEIGHT}")

if device == 'cuda':
    HIDDEN_DIM = 64
    gnn_channels = [128, 256]#[64, 64]
    gnn_channels = [64, 128, 256]
    gnn_channels = [256, 256]
    nn_channels = [64, 32, 8]
    #nn_channels = [64, 16]
    DISJOINT_RATIO = 0.99
    #nn_channels = []
    if MODEL == 'complex':
        node_dim = 128
    else:
        node_dim = 128
else:
    gnn_channels = [2, 2]
    nn_channels = [2]
    node_dim = 2
    HIDDEN_DIM = 4
    EPOCHS = 2
    neg_ratio = 2
    # num_batches = 2
    DISJOINT_RATIO = 0.5
    
MESSAGE_RATIO = DISJOINT_RATIO
print(f"gnn channels: {gnn_channels}")
print(f"nn channels: {nn_channels}")
print(f"node dim: {node_dim}")
print(f"hidden dim: {HIDDEN_DIM}")
print(f"val batch size: {VAL_BATCH_SIZE}")
print(f"model type: {MODEL}")
print(f"disjoint or message ratio: {DISJOINT_RATIO}")
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

NEG_VAL_RATIO = valid_graph[('drug', 'drug-drug_cancer', 'drug')].neg_edge_label_index.shape[2]

# predicted_edge_types = [edge_type for edge_type in train_graph.edge_types if not edge_type[1].startswith('rev_')]
predicted_edge_types = list(set(valid_graph.edge_label_index_dict.keys()).union(set(test_graph.edge_label_index_dict.keys())))

if MODEL == 'classifier':
    model = Classifier(
        gnn_channels=gnn_channels,
        nn_channels=nn_channels,
        embedding_dims={k: node_dim for k in train_graph.node_types},
        number_classes={k: len(v) for k,v in train_graph.node_id_dict.items()},
        message_edge_types=list(train_graph.edge_types),
        predicted_edge_types=predicted_edge_types,
        custom=False,
        device=device
    )
    loss_func = BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
elif MODEL == 'complex':
    model = MultiDomainComplEx(
        gnn_channels=gnn_channels,
        embedding_dims={k: node_dim for k in train_graph.node_types},
        number_classes={k: len(v) for k,v in train_graph.node_id_dict.items()},
        message_edge_types=list(train_graph.edge_types),
        predicted_edge_types=predicted_edge_types,
        hidden_dim=HIDDEN_DIM,
        use_layernorm=True,
        device=device
    )
    loss_func = BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
elif MODEL == 'rotate':
    model = MultiDomainRotatE(
        gnn_channels=gnn_channels,
        embedding_dims={k: node_dim for k in train_graph.node_types},
        number_classes={k: len(v) for k,v in train_graph.node_id_dict.items()},
        message_edge_types=list(train_graph.edge_types),
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
# print(f"num batches: {num_batches}")
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=REGULARIZATION)

ev = Evaluator(name='ogbl-biokg')

valid_graph = valid_graph.to(device)
k = predicted_edge_types[0]
init_edges = {k: train_graph.edge_index_dict[k].to(device)}
with torch.no_grad():
    _ = model(valid_graph, edges_to_predict=init_edges, return_embs=False)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
nbr_params = count_trainable_params(model)
print("Trainable params:", nbr_params)

best_mrr = -np.inf

sem_loss_each_epoch = True
print(f"sem loss each epoch: {sem_loss_each_epoch}")
# %%
edge_idx_lens = {k: train_graph.edge_index_dict[k].shape[1] for k in predicted_edge_types}
if sem_loss_each_epoch:
    train_graph = train_graph.to('cpu')
else:
    train_graph = train_graph.to(device)
since_improved = 0
split_train_graph = copy.copy(train_graph).to(device)
for epoch in tqdm.tqdm(range(EPOCHS)):
    # split_train_graph, _, _ = link_split(train_graph)
    epoch_loss = 0
    epoch_sem_loss = 0
    total_sem_loss = total_neg_sem_loss = 0
    total_epoch_loss = total_epoch_examples = 0
    total_sem_ex = total_neg_sem_ex = 0

    if epoch == 20:
        LR = 5e-2
        for g in optimizer.param_groups:
            g['lr'] = LR
        print(f"updated learning rate: {LR}")
    elif epoch == 100:
        LR = 1e-2
        for g in optimizer.param_groups:
            g['lr'] = LR
        print(f"updated learning rate: {LR}")
    elif epoch == 200:#int(EPOCHS/3):
        LR = LR / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR
        print(f"updated learning rate: {LR}")
    elif epoch == 300:#int(EPOCHS/2):
        LR = LR / 5
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR
        print(f"updated learning rate: {LR}")

    if not sem_loss_each_epoch:
        if SEM_WEIGHT > 0:
            # train_graph = train_graph.to(device)
            optimizer.zero_grad()
            x_dicts = model.get_embeddings(split_train_graph, return_embs=True)
            sem_loss, neg_sem_loss, layer_losses = box_loss(x_dicts,
                                gci0, loss_type='distance',
                                neg=(NEG_WEIGHT > 0), return_layer_loss=True, domain_to_train=GCI0_DOMAINS_TO_TRAIN)
            loss = sem_loss + NEG_WEIGHT * neg_sem_loss
            loss.backward()
            optimizer.step()

            print(f"epoch sem loss: {loss.detach().item()}")

    epoch_layer_losses = []
    
    for edge_indices in get_index_splits(edge_idx_lens, MESSAGE_RATIO):
        for k, (message_edges, label_edges) in edge_indices.items():
            pos_edges = train_graph[k].edge_index[:, label_edges].to(device)
            message_edge_index = train_graph[k].edge_index[:, message_edges]
            rev = (k[2], 'rev_' + k[1], k[0])
            rev_message_edge_index = train_graph[rev].edge_index[:, message_edges]
            head_max_id = node_max_ids[k[0]]
            tail_max_id = node_max_ids[k[2]]
            neg_edges = sample_neg_ex(
                    len(label_edges),
                    head_max_id,
                    tail_max_id,
                    # head=pos_edges[0,:],
                    neg_ratio=neg_ratio,
                    #device=device
                ).to(device)

            split_train_graph[k].edge_index = message_edge_index.to(device)

            split_train_graph[rev].edge_index = rev_message_edge_index.to(device)
            edge_label_indices = torch.cat((pos_edges, neg_edges), dim=1)
            edge_labels = torch.cat((torch.ones_like(pos_edges[0,:], dtype=torch.float), torch.zeros_like(neg_edges[0,:], dtype=torch.float)))


            split_train_graph[k].edge_label_index = edge_label_indices.to(device)
            split_train_graph[k].edge_label = edge_labels.to(device)
        split_train_graph = split_train_graph.to(device)
        optimizer.zero_grad()
        if sem_loss_each_epoch:
            if SEM_WEIGHT > 0:
                # y, x_dicts = model(split_train_graph, edges_to_predict=split_train_graph.edge_label_index_dict, return_embs=True)
                y, x_dicts = model(split_train_graph, edges_to_predict=split_train_graph.edge_label_index_dict, return_embs=True)
                sem_loss, neg_sem_loss, layer_losses, nbr_sem = box_loss(x_dicts,
                                    gci0, loss_type='distance',
                                    neg=(NEG_WEIGHT > 0), return_layer_loss=True, domain_to_train=GCI0_DOMAINS_TO_TRAIN)
                epoch_layer_losses.append(layer_losses)
                total_sem_loss += sem_loss.detach().item()
                total_sem_ex += nbr_sem[0]
                if NEG_WEIGHT > 0:
                    total_neg_sem_loss += neg_sem_loss.detach().item()
                    total_neg_sem_ex += nbr_sem[1]

            else:    
                # y = model(split_train_graph, edges_to_predict=split_train_graph.edge_label_index_dict, return_embs=False)
                y = model(split_train_graph, edges_to_predict=split_train_graph.edge_label_index_dict, return_embs=False)
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
            preds = (y.detach() > 0).type(torch.int).cpu().numpy()
        else:
            loss = loss_func(y.squeeze(), all_labels)
            # preds = sigmoid(y.detach()).cpu().numpy().round()
        epoch_loss += loss.detach().item()
        if sem_loss_each_epoch:
            if SEM_WEIGHT > 0:
                sl = SEM_WEIGHT * (sem_loss + NEG_WEIGHT * neg_sem_loss)
                loss += sl
                epoch_sem_loss += sl.detach().item()
        
        # acc = accuracy_score(all_labels.cpu().numpy(), preds)
        # print(f"train loss: {loss}")
        # print(f"train acc: {acc}")
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.detach().item()
        total_epoch_examples += len(all_labels)

    if device != 'cpu':
        if SEM_WEIGHT > 0 and False:
            del x_dicts, all_labels, y
        else:
            del all_labels, y
        torch.cuda.empty_cache()
    print(f"epoch train loss: {epoch_loss}")
    print(f"epoch sem loss: {epoch_sem_loss}")

    with torch.no_grad():
        val_data = {'y_pred_pos': torch.tensor([], device=device), 'y_pred_neg': torch.tensor([], device=device)}
        for k in valid_graph.edge_label_dict.keys():
            pos = valid_graph[k].pos_edge_label_index
            pos_pred, _ = model(valid_graph, edges_to_predict={k: pos}, return_embs=True)
            val_data['y_pred_pos'] = torch.cat((val_data['y_pred_pos'], pos_pred), dim=0)
            neg = valid_graph[k].neg_edge_label_index
            val_batches = int(np.ceil(neg.shape[1]*neg.shape[2]/VAL_BATCH_SIZE))
            val_size = int(VAL_BATCH_SIZE / neg.shape[2])
            idx = 0
            for i in range(val_batches):
                neg_data = neg[:,idx:idx+val_size,:].reshape((2, -1))
                neg_pred = model(valid_graph, edges_to_predict={k: neg_data}, return_embs=False)
                val_data['y_pred_neg'] = torch.cat((val_data['y_pred_neg'],
                                                neg_pred.reshape((-1,NEG_VAL_RATIO))), dim=0)
                idx += val_size
        val_data['y_pred_pos'] = val_data['y_pred_pos'].squeeze()
        for v in val_data.values():
            print(v.shape)

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

        if device != 'cpu':
            del val_data, pos, neg, neg_data
            torch.cuda.empty_cache()

        if val['mrr_list'].mean() > best_mrr:
            best_mrr = val['mrr_list'].mean()
            print('copying best model...', flush=True)
            best_model = copy.deepcopy(model)
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= 40 and best_mrr > 0.9:
                print('Early stopping...')
                break


print(f"Best MRR: {best_mrr}")
if SEM_WEIGHT > 0:
    boxes = metrics['box_losses']
    keys = boxes[0][0][0]['pos'].keys()
    if NEG_WEIGHT > 0:
        all_boxes = [{pn: {k: [b[-1][l][pn][k] for b in boxes] for k in keys} for pn in ('pos', 'neg')} for l in range(len(boxes[0][0]))]
    else:
        all_boxes = [{'pos': {k: [b[-1][l]['pos'][k] for b in boxes] for k in keys}} for l in range(len(boxes[0][0]))]
    metrics['box_losses'] = all_boxes


# %%
if device != 'cpu':
    del train_graph, valid_graph, model
    torch.cuda.empty_cache()
best_model.to(device)
test_graph = test_graph.to(device)

# VAL_BATCH_SIZE = int(VAL_BATCH_SIZE / 2)
with torch.no_grad():
    test_data = {'y_pred_pos': torch.tensor([], device=device), 'y_pred_neg': torch.tensor([], device=device)}
    for k in test_graph.edge_label_index_dict.keys():
        pos = test_graph[k].pos_edge_label_index
        pos_pred = best_model(test_graph, edges_to_predict={k: pos}, return_embs=False)
        test_data['y_pred_pos'] = torch.cat((test_data['y_pred_pos'], pos_pred), dim=0)
        neg = test_graph[k].neg_edge_label_index
        val_batches = int(np.ceil(neg.shape[1]*neg.shape[2]/VAL_BATCH_SIZE))
        val_size = int(VAL_BATCH_SIZE / neg.shape[2])
        idx = 0
        # print('neg:', neg.shape)
        for i in range(val_batches):
            neg_data = neg[:,idx:idx+val_size].reshape((2, -1))
            neg_pred = best_model(test_graph, edges_to_predict={k: neg_data}, return_embs=False)
            test_data['y_pred_neg'] = torch.cat((test_data['y_pred_neg'],
                                        neg_pred.reshape((-1,NEG_VAL_RATIO))), dim=0)
            idx += val_size
            
    test_data['y_pred_pos'] = test_data['y_pred_pos'].squeeze()
    for k,v in test_data.items():
        print(k, v.shape)

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
fn = os.path.join(BASE, f'results/metrics-biokg/metrics_{time.strftime("%Y%m%d-%H%M%S")}.pkl')
print(f"saving metrics to {fn}...")
with open(fn, 'wb') as fi:
    pickle.dump(metrics, fi)
# %%
