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
    idx = random.sample(range(P.shape[0]), math.ceil((1-p)*P.shape[0]))
    return P[idx,:], F[idx,:] if F is not None else None
    
def voxelgrid(cloud, leaf_size):
    """ Downsamples point cloud """
    vg = cloud.make_voxel_grid_filter()
    vg.set_leaf_size(leaf_size, leaf_size, leaf_size)       
    return vg.filter()    
    
    
    
def create_graph(cloud, knn, radius, kd=None):
    xyz = cloud.to_array()[:,:3]   
    num_nodes = xyz.shape[0]
    
    kd = cloud.make_kdtree_flann() if kd is None else kd
    if knn:
        indices, sqr_distances = kd.nearest_k_search_for_cloud(cloud, radius)
    else:
        indices, sqr_distances = kd.radius_search_for_cloud(cloud, radius, RADIUS_MAX_K)
        
    edges = []
    offsets = []
    for i in range(num_nodes):    
        for j in range(indices.shape[1]): #includes self-loops
            idx = indices[i][j]
            if j>0 and idx==0 and sqr_distances[i][j]<1e-8: continue
            edges.append((idx,i))
            offsets.append(xyz[i] - xyz[idx])
            
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
        #print(cloud, len(graph.nodes()), len(graph.edges()))        
        prev_res = res

    return graphs, pooldata

        
# pool: node labeling can be independent in graphs. just create mapping.
# conv-stride: gets 2 clouds, returns labels for the second
# conv: gets 1 cloud, 






# pooling: creates list of lists   ... or it will be just a special function to pass to GraphPoolInfo?

# needs: leafsz, reach = list of pairs. leafsz changes -> mp. only reach changes -> conv

# batch merging function - gi

# model: how does it assign info->conv, what belongs to what? As before: generate the list of unique tuples, give it to create_graph.

# c_rad_16
# m_res_(rad):  voxelgrid,mp ... but what reach


# c_res_rad_16
# m_res_rad
# ... can get only smaller
# ... keeps up-to-date pair, once updated, append to list of pairs and notes id to assign Info later
# -> here I get the list of pairs. leafsz changes -> pooling_map (or later strided_conv). only reach changes -> cloud_to_graph.

# merging: 





### like in paper:
#initial_res_rad
#c                          [c_res_rad for strided one]
#m_res_rad
#g_rad


