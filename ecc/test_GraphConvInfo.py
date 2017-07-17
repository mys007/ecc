import unittest
import numpy as np
import networkx as nx
import torch

from .GraphConvInfo import GraphConvInfo, EdgeFeatureCache




class TestGraphConvInfo(unittest.TestCase):
    
    def test_edgefeat_full(self):
    
        def my_edge_features(edges):
            #feat = []
            #for e in edges:
            #    e['d'] if e['d'] else torch.Tensor([0 0])
            #    feat.append()
            feat = [ (e['d'] if 'd' in e else torch.Tensor([[0, 0]])) for e in edges ]
            feat = torch.cat(feat, 0)
            return feat, None    
    
        G = nx.DiGraph()
        G.add_nodes_from([1,4,3])
        G.add_edges_from([(1,4,{'d':torch.Tensor([[11, 14]])}), 
                          (3,4,{'d':torch.Tensor([[13, 14]])}),
                          (3,3,{'d':torch.Tensor([[33, 33]])})])

        GI = GraphConvInfo()
        GI.set_batch([G], my_edge_features)
    
        GI2 = GraphConvInfo()
        GI2.set_batch([G], None, EdgeFeatureCache([G],my_edge_features))             

        self.assertTrue((GI._edgefeats - GI2._edgefeats).abs().sum()==0) 
    
    
    def test_edgefeat_sparse(self):
        def my_edge_features(edges):
            feat = [ e['d'] if 'd' in e else -1 for e in edges ]
            ufeat, indices = np.unique(np.array(feat), return_inverse=True) #for whole rows: https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
            return torch.from_numpy(ufeat.astype(np.float32)), torch.from_numpy(indices)

        G = nx.DiGraph()
        G.add_nodes_from([3,2,1])
        G.add_edges_from([(1,4,{'d':2}), 
                          (3,4,{'d':3}),
                          (3,3,{'d':3})])

        GI = GraphConvInfo()
        GI.set_batch([G], my_edge_features)
        
        GI2 = GraphConvInfo()
        GI2.set_batch([G], None, EdgeFeatureCache([G],my_edge_features))             

        self.assertTrue((GI._edgefeats - GI2._edgefeats).abs().sum()==0)
        self.assertTrue((GI._idxe - GI2._idxe).abs().sum()==0)
        
        
            
if __name__ == '__main__':
    unittest.main()