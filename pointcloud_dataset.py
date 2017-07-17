from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import math
import transforms3d

import torch
import torchnet as tnt
import networkx as nx

import pcl
import pointcloud_utils as pcu

import ecc


SYDNEY_PATH = './datasets/sydney/'



def cloud_edge_feats(edges, args):
    columns = []
    offsets = np.asarray([ e['d'] for e in edges ])
    
    # todo: possible discretization, round to multiples of min(offsets[offsets>0]) ?
    
    if 'eukl' in args.pc_attribs:
        columns.append(offsets)
    
    if 'polar' in args.pc_attribs:
        p1 = np.linalg.norm(offsets, axis=1)
        p2 = np.arctan2(offsets[:,1], offsets[:,0])
        p3 = np.arccos(offsets[:,2] / (p1 + 1e-6))
        columns.extend([p1[:,np.newaxis], p2[:,np.newaxis], p3[:,np.newaxis]])

    edgefeats = np.concatenate(columns, axis=1).astype(np.float32)
    
    if args.edgecompaction:
        edgefeats_clust, indices = ecc.unique_rows(edgefeats)
        print('Edge features: {} -> {} unique edges, {} dims'.format(edgefeats.shape[0], edgefeats_clust.shape[0], edgefeats_clust.shape[1]))
        return torch.from_numpy(edgefeats_clust), torch.from_numpy(indices)
    else:
        print('Edge features: {} edges, {} dims'.format(edgefeats.shape[0], edgefeats.shape[1]))
        return torch.from_numpy(edgefeats), None
        

        
        
def sydney_dataset(args, pyramid_conf, training):

    names = ['t','intensity','id', 'x','y','z', 'azimuth','range','pid']
    formats = ['int64', 'uint8', 'uint8', 'float32', 'float32', 'float32', 'float32', 'float32', 'int32']
    binType = np.dtype( dict(names=names, formats=formats) ) # official read-bin.py from sydney
    
    classmap = {'4wd':0, 'building':1, 'bus':2, 'car':3, 'pedestrian':4, 'pillar':5, 'pole':6, 'traffic_lights':7, 
                'traffic_sign':8, 'tree':9, 'truck':10, 'trunk':11, 'ute':12, 'van':13}

    def loader(filename):
        data = np.fromfile(filename, binType)
        cls = classmap[os.path.basename(filename).split('.')[0]] 
        P = np.vstack([data['x'], data['y'], data['z']]).T # metric units
        F = data['intensity'].reshape(-1,1)

        # training data augmentation
        if training:
            if args.pc_augm_input_dropout > 0: # removing points here changes graph structure (unlike zeroing features)
                P, F = pcu.dropout(P, F, args.pc_augm_input_dropout)
                
            M = np.eye(3)
            if args.pc_augm_scale > 1:
                s = random.uniform(1/args.pc_augm_scale, args.pc_augm_scale)
                M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
            if args.pc_augm_rot:
                angle = random.uniform(0, 2*math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], angle), M) # z=upright assumption
            if args.pc_augm_mirror_prob > 0: # mirroring x&y, not z
                if random.random() < args.pc_augm_mirror_prob/2:
                    M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
                if random.random() < args.pc_augm_mirror_prob/2:
                    M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,1,0]), M)
                
            P = np.dot(P, M.T)
  
        # coarsen to initial resolution (btw, axis-aligned quantization of rigidly transformed cloud adds jittering noise)    
        PF = np.hstack([P, F]).astype(np.float32)
        PF_filtered = pcu.voxelgrid(pcl.PointCloud_PointXYZI(PF), pyramid_conf[0][0]).to_array() # aggregates intensities too (note pcl wrapper bug: only int intensities accepted)
        F = PF_filtered[:,3]/255 - 0.5 # laser return intensities in [-0.5,0.5]

        cloud = pcl.PointCloud(PF_filtered[:,0:3]) # (pcl wrapper bug: XYZI cannot query kd-tree by radius)
        graphs, poolmaps = pcu.create_graph_pyramid(args, cloud, pyramid_conf)     

        return F, cls, graphs, poolmaps
            
    def create_dataset(foldnr):
        return tnt.dataset.ListDataset('{}/folds/fold{:d}.txt'.format(SYDNEY_PATH, foldnr), loader, SYDNEY_PATH + '/objects')

    if training:
        datasets = []
        for f in range(0,4):
            if f != args.cvfold:
                datasets.append(create_dataset(f))
        return tnt.dataset.ConcatDataset(datasets)
    else:
        return create_dataset(args.cvfold)
        
        
    # todo: should rewrite as class, so that I can set `manualSeed` in __getstate__   (multiprocessing should set seed, set in the dataset)












