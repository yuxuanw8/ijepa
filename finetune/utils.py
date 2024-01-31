import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
import math
import random
import time

def evaluate(model, data_loader, binary=False, normalize=None,
    show_progress=False, device=None):
    """
    Computes predictions, accuracy and loss (optional) of model on 
    the given data loader.

    Args:
        model: the model to compute accuracy for.
        data_loader: the data loader to compute accuracy over.
        loss_fn: the loss function to use.
        binary: whether the model is a binary classifier.
        normalize: Normalization to apply to the data (e.g. if data_loader did
            not already normalize).
        show_progress: whether to show a progress bar.
        device: the device to load the data onto.

    Returns:
        The accuracy of the model on the data.
    """
    correct = torch.tensor(0, dtype=torch.int, device=device)
    total_items = torch.tensor(0, dtype=torch.int, device=device)
    predictions = []
    probabilities = []

    loss_fn = nn.CrossEntropyLoss()

    if loss_fn is not None:
        total_xe_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    else:
        total_xe_loss = None

    with torch.no_grad():
        progress = data_loader

        if show_progress:
            progress = tqdm(progress, leave=False, desc="Batches",
                dynamic_ncols=True)

        for data in progress:
            if device is None:
                inputs, labels = data
            else:
                inputs, labels = data[0].to(device), data[1].to(device)

            if normalize is not None:
                inputs = normalize(inputs)

            outputs = model(inputs)

            if binary:
                outputs = torch.squeeze(outputs, -1)
                outputs = F.sigmoid(outputs)
                probs = outputs
            else:
                probs = F.softmax(outputs, -1)
                
            probabilities.append(probs)

            if loss_fn is not None:
                xe_loss = loss_fn(outputs, labels)
                total_xe_loss += len(labels) * xe_loss
            
            if binary:
                predicted_classes = (probs > 0.5).long()
            else:
                _, predicted_classes = torch.max(outputs.data, 1)
            
            predictions.append(predicted_classes)
            
            total_items += len(labels)
            correct += (predicted_classes == labels).sum()
    
    
    if len(predictions) > 0:
        predictions = torch.concat(predictions)
    
    if len(probabilities) > 0:
        probabilities = torch.concat(probabilities)
    
    return predictions, probabilities, correct, total_xe_loss, total_items

import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    """
    hard-code to 32
    TODO: fix the model.blocks raising the error 
    """
    num_layers = len(model.blocks) + 1
    #num_layers = 32 + 1

    """
    in our case, num_layers=33, so layer_scales has 34 elements from 0 to 33.
    """
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    Also, note, in I-JEPA, it does not use cls token in ViT, but that does not affect
    anything, since 0 is assigned to anything before the blocks (cls_token, patch_embed,
    pos_embed) and 1 ~ 32 is assigned to blocks indexed by the block number, and 33 is 
    assigned to anything after blocks, num_layers=33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers