"""
    Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs
    https://github.com/mys007/ecc
    https://arxiv.org/abs/1704.02901
    2017 Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import igraph
import torch

    
class GraphConvInfo(object):          
    """ Holds information about the structure of graph(s) in a vectorized form useful to `GraphConvModule`. 
    
    We assume that the node feature tensor (given to `GraphConvModule` as input) is ordered by igraph vertex id, e.g. the fifth row corresponds to vertex with id=4. Batch processing is realized by concatenating all graphs into a large graph of disconnected components (and all node feature tensors into a large tensor).

    The class requires problem-specific `edge_feat_func` function, which receives list of edge attributes (dict) and returns Tensor of edge features and LongTensor of inverse indices if edge compaction was performed (less unique edge features than edges so some may be reused).
    """

    def __init__(self, *args, **kwargs):
        self._idxn = None           #indices into input tensor of convolution (node features)
        self._idxe = None           #indices into edge features tensor (or None if it would be linear, i.e. no compaction)
        self._degrees = None        #in-degrees of output nodes (slices _idxn and _idxe)
        self._degrees_gpu = None
        self._edgefeats = None      #edge features tensor (to be processed by feature-generating network)
        if len(args)>0 or len(kwargs)>0:
            self.set_batch(*args, **kwargs)
      
    def set_batch(self, graphs, edge_feat_func):
        """ Creates a representation of a given batch of graphs.
        
        Parameters:
        graphs: single graph or a list/tuple of graphs. The graphs are supposed to be igraph objects with self-loops (!).
        edge_feat_func: see class description.
        """
        
        graphs = graphs if isinstance(graphs,(list,tuple)) else [graphs]
        p = 0
        idxn = []
        degrees = []
        edges = []
                
        for i,G in enumerate(graphs):
        
            indeg = G.indegree(G.vs, loops=True)
            
            for v in range(G.vcount()):               
                for u in G.predecessors(v): # we assume that self-loops are in the graphs already
                    idxn.append(u+p)
                    edges.append(G.es[G.get_eid(u,v)].attributes())
                    
                degrees.append(indeg[v])        
                
            p += G.vcount()

        self._edgefeats, self._idxe = edge_feat_func(edges)
        
        self._idxn = torch.LongTensor(idxn)
        if self._idxe is not None:
            assert self._idxe.numel() == self._idxn.numel()
            
        self._degrees = torch.LongTensor(degrees)
        self._degrees_gpu = None            
        
    def cuda(self):
        self._idxn = self._idxn.cuda()
        if self._idxe is not None: self._idxe = self._idxe.cuda()
        self._degrees_gpu = self._degrees.cuda()
        self._edgefeats = self._edgefeats.cuda()        
        
    def get_buffers(self):
        """ Provides data to `GraphConvModule`.
        """
        return self._idxn, self._idxe, self._degrees, self._degrees_gpu, self._edgefeats
