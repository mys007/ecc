from __future__ import division
from __future__ import print_function
from builtins import range

import torch


class GraphPoolInfo(object):          

    def __init__(self, *args, **kwargs):
        self._idxn = None           #index into source node features
        self._degrees = None        #in-degrees of output nodes
        if len(args)>0 or len(kwargs)>0:
            self.set_batch(*args, **kwargs)
            
    def set_batch(self, poolmaps, graphs_from, graphs_to):
        poolmaps = poolmaps if isinstance(poolmaps,(list,tuple)) else [poolmaps]
        graphs_from = graphs_from if isinstance(graphs_from,(list,tuple)) else [graphs_from]
        graphs_to = graphs_to if isinstance(graphs_to,(list,tuple)) else [graphs_to]
        
        idxn = []
        self._degrees = []   
        p = 0        
              
        for map, G_from, G_to in zip(poolmaps, graphs_from, graphs_to):
            nodes = G_to.vs
            for node in nodes:
                nlist = map.get(node.index, [])
                idxn.extend([n+p for n in nlist])
                self._degrees.append(len(nlist))
            p += G_from.vcount()
         
        self._idxn = torch.LongTensor(idxn)                        
        
    def get_buffers(self, cuda):
        return self._idxn.cuda() if cuda else self._idxn, self._degrees
 
        
    # def serialize(self):
        # return [self._idxn, self._idxd, self._idxe, torch.LongTensor(self._degrees), self._edgefeats, self._edges_degnorm]
   
    # def deserialize(self, data):
        # self._idxn, self._idxd, self._idxe, self._degrees, self._edgefeats, self._edges_degnorm = data
        # self._degrees = list(self._degrees)
