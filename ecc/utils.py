from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import torch
import networkx as nx

import ecc

def graph_info_collate_classification(batch, edge_func):
    features, classes, graphs, pooldata = list(zip(*batch))
    graphs_by_layer = list(zip(*graphs))
    pooldata_by_layer = list(zip(*pooldata))
    
    features = torch.cat([torch.from_numpy(f) for f in features])
    classes = torch.LongTensor(classes)
    
    GIs, PIs = [], []    
    for graphs in graphs_by_layer:
        GIs.append( ecc.GraphConvInfo(graphs, edge_func) )
    for pooldata in pooldata_by_layer:
        PIs.append( ecc.GraphPoolInfo(*zip(*pooldata)) )  
       
    return features, classes, GIs, PIs
    
    
def unique_rows(data):
    # https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    uniq, indices = np.unique(data.view(data.dtype.descr * data.shape[1]), return_inverse=True)
    return uniq.view(data.dtype).reshape(-1, data.shape[1]), indices
    
def one_hot_discretization(feat, clip_min, clip_max, upweight):
    indices = np.clip(np.round(feat), clip_min, clip_max).astype(int).reshape((-1,))
    onehot = np.zeros((feat.shape[0], clip_max - clip_min + 1))
    onehot[np.arange(onehot.shape[0]), indices] = onehot.shape[1] if upweight else 1
    return onehot    