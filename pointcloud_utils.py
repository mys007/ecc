"""
    Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs
    https://github.com/mys007/ecc
    https://arxiv.org/abs/1704.02901
    2017 Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import time
import random
import numpy as np
import math
import transforms3d
import pcl
import igraph
from collections import defaultdict

RADIUS_MAX_K = 100

def dropout(P,F,p):
    """ Removes points with probability p from vector of points and features"""
    idx = random.sample(range(P.shape[0]), int(math.ceil((1-p)*P.shape[0])))
    return P[idx,:], F[idx,:] if F is not None else None
    
def voxelgrid(cloud, leaf_size):
    """ Downsamples point cloud """
    vg = cloud.make_voxel_grid_filter()
    vg.set_leaf_size(leaf_size, leaf_size, leaf_size)       
    return vg.filter()    
    
     
def create_graph(cloud, knn, radius, kd=None):
    """ Converts point cloud to graph by building neighborhood structure using kd-tree.
    Parameters:
    knn: true if using k-nn neighbors, false if using radius neighbors
    radius: radius if not knn, k if knn
    """
    xyz = cloud.to_array()[:,:3]   
    num_nodes = xyz.shape[0]

    kd = cloud.make_kdtree_flann() if kd is None else kd
    if knn:
        indices, sqr_distances = kd.nearest_k_search_for_cloud(cloud, radius)
    else:
        indices, sqr_distances = kd.radius_search_for_cloud(cloud, radius, RADIUS_MAX_K)    
        
    sqr_distances[:,0] += 1 #includes self-loops
    valid = np.logical_or(indices > 0, sqr_distances>1e-10)
    rowi, coli = np.nonzero(valid)
    idx = indices[(rowi,coli)]
    
    edges = np.vstack([idx, rowi]).T.tolist()
    offsets = xyz[rowi] - xyz[idx]
  
    G = igraph.Graph(n=num_nodes, edges=edges, directed=True, edge_attrs={'offset':offsets})
    return G, kd         
    

def create_pooling_map(cloud_src, cloud_dst):
    """ Creates a pooling map from cloud_dst to cloud_src. Points in the denser cloud (source) are mapped to their nearest neighbor in the subsampled cloud (dest), thus no overlaps supported. """
    kd_dst = cloud_dst.make_kdtree_flann()
    indices, _ = kd_dst.nearest_k_search_for_cloud(cloud_src, 1)
    
    poolmap = defaultdict(list)
    for si, di in enumerate(indices):
        poolmap[di[0]].append(si) 
    
    return poolmap, kd_dst
     
     
def create_graph_pyramid(args, cloud, pyramid_conf):
    """ Builds a pyramid of graphs and pooling operations corresponding to progressively coarsened point cloud using voxelgrid.
    Parameters:
    pyramid_conf: list of tuples (grid resolution, neigborhood radius/k), defines the pyramid
    """
    
    graphs = []
    pooldata = []
    prev_res = pyramid_conf[0][0] # assuming the caller performed the initial sampling.
    kd = None
    
    for res, rad in pyramid_conf:
        if prev_res != res:
            cloud_new = voxelgrid(cloud, res)
            poolmap, kd = create_pooling_map(cloud, cloud_new)
            cloud = cloud_new

        graph, kd = create_graph(cloud, args.pc_knn, rad, kd)    
        graphs.append(graph)   
        if prev_res != res:
            pooldata.append((poolmap, graphs[-2], graphs[-1]))
        prev_res = res

    return graphs, pooldata
