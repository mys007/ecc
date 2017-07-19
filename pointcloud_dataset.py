from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import math
import transforms3d
import functools
import logging

import torch
import torchnet as tnt

import pcl
import pointcloud_utils as pcu

import ecc


SYDNEY_PATH = './datasets/sydney/'
MODELNET10_PATH = './datasets/modelnet10/'
MODELNET40_PATH = './datasets/modelnet40/'


def cloud_edge_feats(edges, args):
    columns = []
    offsets = np.asarray([ e['offset'] for e in edges ])
    
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
        logging.debug('Edge features: %d -> %d unique edges, %d dims', edgefeats.shape[0], edgefeats_clust.shape[0], edgefeats_clust.shape[1])
        return torch.from_numpy(edgefeats_clust), torch.from_numpy(indices)
    else:
        logging.debug('Edge features: %d edges, %d dims',edgefeats.shape[0], edgefeats.shape[1])
        return torch.from_numpy(edgefeats), None
        

        
        
        
        
def get_sydney_info(args):
    return {
        'feats': 1,
        'edge_feats': (3 if 'eukl' in args.pc_attribs else 0) + (3 if 'polar' in args.pc_attribs else 0),
        'classes': 14,
        'test_set_expansion': 1,
    }
        
        
def get_sydney(args, pyramid_conf, training):

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
        P -= np.min(P,axis=0) #move to positive octant (voxelgrid has fixed boundaries at axes planes)
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
        for f in range(4):
            if f != args.cvfold:
                datasets.append(create_dataset(f))
        return tnt.dataset.ConcatDataset(datasets)           
        
    else:
        return create_dataset(args.cvfold)
     


def get_modelnet_info(args):
    return {
        'feats': 1,
        'edge_feats': (3 if 'eukl' in args.pc_attribs else 0) + (3 if 'polar' in args.pc_attribs else 0),
        'classes': 10 if args.dataset=='modelnet10' else 40,
        'test_set_expansion': 12, #over orientations
    }
    

def get_modelnet(args, pyramid_conf, training):
    if args.dataset=='modelnet10':
        path = MODELNET10_PATH
        classmap = {'bathtub':0, 'bed':1, 'chair':2, 'desk':3, 'dresser':4, 'monitor':5, 'night_stand':6, 'sofa':7, 'table':8, 'toilet':9}
    else:
        path = MODELNET40_PATH
        classmap = {'airplane':0, 'bathtub':1, 'bed':2, 'bench':3, 'bookshelf':4, 'bottle':5, 'bowl':6, 'car':7, 'chair':8, 'cone':9, 'cup':10, 
                    'curtain':11, 'desk':12, 'door':13, 'dresser':14, 'flower_pot':15, 'glass_box':16, 'guitar':17, 'keyboard':18, 'lamp':19, 
                    'laptop':20, 'mantel':21, 'monitor':22, 'night_stand':23, 'person':24, 'piano':25, 'plant':26, 'radio':27, 'range_hood':28, 
                    'sink':29, 'sofa':30, 'stairs':31, 'stool':32, 'table':33, 'tent':34, 'toilet':35, 'tv_stand':36, 'vase':37, 'wardrobe':38, 'xbox':39}    

    def loader(filename, test_angle=None):
        P = pcl.load(filename).to_array()
        cls = classmap['_'.join(os.path.basename(filename).split('_')[:-1])] 
        F = np.zeros((P.shape[0],1), dtype=np.float32) # no point features in modelnet
        
        #transform into ball of diameter 32 (obj scale in modelnet has no meaning, original meshes have random sizes) 
        # (in the paper we used a unit ball and ./32 grid sizes, this is equivalent in effect)
        diameter = np.max(np.max(P,axis=0) - np.min(P,axis=0))
        M = transforms3d.zooms.zfdir2mat(32/diameter)      

        # training data augmentation
        if training:
            if args.pc_augm_input_dropout > 0: # removing points here changes graph structure (unlike zeroing features)
                P, _ = pcu.dropout(P, None, args.pc_augm_input_dropout)
                
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
        else:
            if test_angle:
                M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], test_angle), M) # z=upright assumption
                
        P = np.dot(P, M.T)
  
        # coarsen to initial resolution (btw, axis-aligned quantization of rigidly transformed cloud adds jittering noise)
        P -= np.min(P,axis=0) #move to positive octant (voxelgrid has fixed boundaries at axes planes)
        cloud = pcu.voxelgrid(pcl.PointCloud(P.astype(np.float32)), pyramid_conf[0][0])

        graphs, poolmaps = pcu.create_graph_pyramid(args, cloud, pyramid_conf)     

        return F, cls, graphs, poolmaps
            
    def create_dataset(test_angle=None):
        ploader = functools.partial(loader, test_angle=test_angle)
        return tnt.dataset.ListDataset('{}/filelist_{}.txt'.format(path, 'train' if training else 'test'), ploader, path + '/pc_sampled1k')

    if training:
        return create_dataset()
    else:
        datasets = []
        for a in range(12):
            datasets.append(create_dataset(a/12 * 2*math.pi))
        concated = tnt.dataset.ConcatDataset(datasets)      
        # reshuffle to put same samples with different rotation after each other for easy aggregation afterwards      
        return tnt.dataset.ResampleDataset(concated, lambda d,i: int((i%12) * len(d)/12 + math.floor(i/12)))           
        
 


