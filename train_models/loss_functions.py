import torch as th
from box_embeddings.parameterizations import MinDeltaBoxTensor, SigmoidBoxTensor#TanhBoxTensor
from box_embeddings.modules.intersection import GumbelIntersection, HardIntersection
from box_embeddings.modules.volume import BesselApproxVolume, HardVolume

def box_loss(embeddings, gci0, loss_type='inclusion', box_transform='mindelta',
             inter='gumbel', inter_temp=0.1, vol='bessel', vol_temp=0.1,
             gamma=0.0, neg=False, return_layer_loss=False, domain_to_train=[],
             **kwargs):
    match box_transform:
        case 'mindelta':
            box = MinDeltaBoxTensor
        case 'sigmoid':
            box = SigmoidBoxTensor
        case _:
            raise NotImplementedError()
    if loss_type == 'inclusion':
        return box_loss_inclusion(embeddings, gci0, box=box, inter=inter,
                                  inter_temp=inter_temp, vol=vol,
                                  vol_temp=vol_temp, neg=neg,
                                  return_layer_loss=return_layer_loss)
    if loss_type == 'distance':
        return box_loss_distance(embeddings, gci0, box=box, gamma=gamma,
                                 neg=neg, return_layer_loss=return_layer_loss,
                                 domain_to_train=domain_to_train)
    pass

def box_loss_inclusion(embeddings, gci0, box=MinDeltaBoxTensor, inter='gumbel',
             inter_temp=0.1, vol='bessel', vol_temp=0.1, neg=False,
             return_layer_loss=False, **kwargs):
    def neg_loss_func(A, B, volume, intersect):
        return (1 - (volume(intersect(A, B)) /
                     th.minimum(volume(A), volume(B)))).clamp(min=1e-9,
                                                              max=1).log().sum()
    # if neg:
    #     raise NotImplementedError("Negative loss not yet implemented for inclusion loss")
    match inter:
        case 'gumbel':
            intersect = GumbelIntersection(intersection_temperature=inter_temp)
        case 'hard':
            intersect = HardIntersection()
        case _:
            raise NotImplementedError()
        
    match vol:
        case 'bessel':
            volume = BesselApproxVolume(intersection_temperature=inter_temp,
                                        volume_temperature=vol_temp, log_scale=False)
        case 'hard':
            volume = HardVolume()
    loss = 0
    neg_loss = 0
    layer_losses = []
    for x_dict in embeddings:
        layer_loss = {}
        neg_layer_loss = {}
        for k, emb in x_dict.items():
            
            if k == 'genes':
                continue
            box_emb = box.from_vector(emb)
            
            subclasses = box_emb[gci0[k][:,0], ...]
            supclasses = box_emb[gci0[k][:,1], ...]

            l = -(volume(intersect(subclasses, supclasses)) / volume(subclasses)).clamp(min=1e-9, max=1).log().sum()
            loss += l
            layer_loss[k] = l.detach().item()

            if neg:
                max_i = len(emb)
                rand_classes = th.randint(low=0, high=max_i,
                                             size=(len(gci0[k]),),
                                             device=gci0[k].device)
                A = box_emb[rand_classes, ...]
                nl = -neg_loss_func(A, supclasses, volume, intersect)
                neg_loss += nl
                neg_layer_loss[k] = nl.detach().item()

                rand_classes = th.randint(low=0, high=max_i,
                                             size=(len(gci0[k]),),
                                             device=gci0[k].device)
                A = box_emb[rand_classes, ...]
                nl = -neg_loss_func(A, subclasses, volume, intersect)
                neg_loss += nl
                neg_layer_loss[k] += nl.detach().item()

                rand_classes = th.randint(low=0, high=max_i,
                                             size=(len(gci0[k]),2),
                                             device=gci0[k].device)
                A = box_emb[rand_classes[:,0], ...]
                B = box_emb[rand_classes[:,1], ...]
                nl = -neg_loss_func(A, B, volume, intersect)
                neg_loss += nl
                neg_layer_loss[k] += nl.detach().item()
            
        layer_losses.append({'pos': layer_loss, 'neg': neg_layer_loss})
    if return_layer_loss:
        return loss, neg_loss, layer_losses
    return loss, neg_loss


def box_loss_distance(embeddings, gci0, box=MinDeltaBoxTensor, gamma=0.0,
                      neg=False, return_layer_loss=False, domain_to_train=[]):

    def dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False):
        if neg:
            v = th.relu(-th.abs(sub_c - sup_c) + sub_o + sup_o + gamma)
            delta = (v > 0).all(dim=-1).float()
            return (delta * v.norm(dim=-1)).sum()
        else:
            return th.relu(th.abs(sub_c - sup_c) + sub_o - sup_o -
                          gamma).norm(dim=-1).sum()
    loss = 0
    neg_loss = 0
    layer_losses = []
    nbr_ex = nbr_neg_ex = 0
    
    for x_dict in embeddings:
        layer_loss = {}
        neg_layer_loss = {}
        for k, emb in x_dict.items():
            if len(domain_to_train) > 0 and k not in domain_to_train:
                continue
            if k == 'genes':
                continue
            box_emb = box.from_vector(emb)

            nbr_ex += len(gci0[k])
            
            subclasses = box_emb[gci0[k][:,0], ...]
            sub_c, sub_o = subclasses.centre, subclasses.Z - subclasses.centre
            supclasses = box_emb[gci0[k][:,1], ...]
            sup_c, sup_o = supclasses.centre, supclasses.Z - supclasses.centre

            l = dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False)
            loss += l
            layer_loss[k] = l.detach().item()
            
            if neg:
                nbr_neg_ex += len(gci0[k]) * 3
                max_i = len(emb)

                rand_classes = th.randint(low=0, high=max_i, size=(len(gci0[k]),), device=gci0[k].device)
                nsub = box_emb[rand_classes, ...]
                nsub_c, nsub_o = nsub.centre, nsub.Z - nsub.centre
                nl = dist_inclusion(nsub_c, nsub_o, sup_c, sup_o, neg=True)
                neg_loss += nl
                neg_layer_loss[k] = nl.detach().item()

                rand_classes = th.randint(low=0, high=max_i, size=(len(gci0[k]),), device=gci0[k].device)
                nsup = box_emb[rand_classes, ...]
                nsup_c, nsup_o = nsup.centre, nsup.Z - nsup.centre
                nl = dist_inclusion(sub_c, sub_o, nsup_c, nsup_o, neg=True)
                neg_loss += nl
                neg_layer_loss[k] += nl.detach().item()

                rand_classes = th.randint(low=0, high=max_i, size=(len(gci0[k]),2), device=gci0[k].device)
                nsub = box_emb[rand_classes[:,0], ...]
                nsub_c, nsub_o = nsub.centre, nsub.Z - nsub.centre
                nsup = box_emb[rand_classes[:,1], ...]
                nsup_c, nsup_o = nsup.centre, nsup.Z - nsup.centre
                nl = dist_inclusion(nsub_c, nsub_o, nsup_c, nsup_o, neg=True)
                neg_loss += nl
                neg_layer_loss[k] += nl.detach().item()

        layer_losses.append({'pos': layer_loss, 'neg': neg_layer_loss})

    if return_layer_loss:
        return loss, neg_loss, layer_losses, (nbr_ex, nbr_neg_ex)
    return loss, neg_loss