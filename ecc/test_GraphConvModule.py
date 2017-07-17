import unittest
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.autograd import Variable, gradcheck

from .GraphConvModule import *
from .GraphConvInfo import GraphConvInfo


class TestGraphConvModule(unittest.TestCase):

    def test_gradcheck(self):

        torch.set_default_tensor_type('torch.DoubleTensor') #necessary for proper numerical gradient
    
        for it1 in range(0,2):
            # without idxe
            n,e,in_channels, out_channels = 20,50,10, 15
            input = torch.randn(n,in_channels)
            weights = torch.randn(e,in_channels,out_channels)
            idxn = torch.from_numpy(np.random.randint(n,size=e))
            idxe = None
            degs = [5, 0, 15, 20, 10]  #strided conv
            edges_degnorm = torch.randn(e) if it1==0 else None
            edge_mem_limit = 30 # some nodes will be combined, some not
            
            func = GraphConvFunction(in_channels, out_channels, idxn, idxe, degs, edges_degnorm, edge_mem_limit=edge_mem_limit)
            data = (Variable(input, requires_grad=True), Variable(weights, requires_grad=True))

            ok = gradcheck(func, data)
            self.assertTrue(ok)
            
            # with idxe
            weights = torch.randn(30,in_channels,out_channels)
            idxe = torch.from_numpy(np.random.randint(30,size=e))

            func = GraphConvFunction(in_channels, out_channels, idxn, idxe, degs, edges_degnorm, edge_mem_limit=edge_mem_limit)

            ok = gradcheck(func, data)
            self.assertTrue(ok)
        
        torch.set_default_tensor_type('torch.FloatTensor')
        
    def xtest_gradcheck_diag(self):

        torch.set_default_tensor_type('torch.DoubleTensor') #necessary for proper numerical gradient
    
        for it1 in range(0,2):
            # without idxe
            n,e,in_channels, out_channels = 20,50,10, 10
            input = torch.randn(n,in_channels)
            weights = torch.randn(e,in_channels)
            idxn = torch.from_numpy(np.random.randint(n,size=e))
            idxe = None
            degs = [5, 0, 15, 20, 10]  #strided conv
            edges_degnorm = torch.randn(e) if it1==0 else None
            edge_mem_limit = 30 # some nodes will be combined, some not
            
            func = GraphConvFunction(in_channels, out_channels, idxn, idxe, degs, edges_degnorm, edge_mem_limit=edge_mem_limit)
            data = (Variable(input, requires_grad=True), Variable(weights, requires_grad=True))

            ok = gradcheck(func, data)
            self.assertTrue(ok)
            
            # with idxe
            weights = torch.randn(30,in_channels,out_channels)
            idxe = torch.from_numpy(np.random.randint(30,size=e))

            func = GraphConvFunction(in_channels, out_channels, idxn, idxe, degs, edges_degnorm, edge_mem_limit=edge_mem_limit)

            ok = gradcheck(func, data)
            self.assertTrue(ok)
        
        torch.set_default_tensor_type('torch.FloatTensor')        
        
        
    def xtest_batch_splitting(self):
    
        n,e,in_channels, out_channels = 20,50,10, 15
        input = torch.randn(n,in_channels)
        weights = torch.randn(e,in_channels,out_channels)
        idxn = torch.from_numpy(np.random.randint(n,size=e))
        idxe = None
        degs = [5, 0, 15, 20, 10]  #strided conv
        
        func = GraphConvFunction(in_channels, out_channels, idxn, idxe, degs, edge_mem_limit=1e10)
        data = (Variable(input, requires_grad=True), Variable(weights, requires_grad=True))
        output1 = func(*data)

        func = GraphConvFunction(in_channels, out_channels, idxn, idxe, degs, edge_mem_limit=1)
        output2 = func(*data)

        self.assertLess((output1-output2).norm(), 1e-6)

    def xtest_edgefeat_full(self):
    
        def my_edge_features(edges):
            feat = [ (e['d'] if 'd' in e else torch.Tensor([[0, 0]])) for e in edges ]
            feat = torch.cat(feat, 0)
            return feat, None    
    
        nodefeat = torch.Tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
    
        G = nx.DiGraph()
        G.add_nodes_from([0,1,2,3,4])
        G.add_edges_from([(1,4,{'d':torch.Tensor([[11, 14]])}), 
                          (4,3,{'d':torch.Tensor([[14, 13]])}),
                          (0,2,{'d':torch.Tensor([[10, 12]])})])
                          
        Gs = [nx.subgraph(G, [0,1,2]), nx.subgraph(G, [4,3,2])]
        print(Gs[0].edges(), Gs[1].edges())
                          

        GI = GraphConvInfo()
        GI.set_batch(Gs, my_edge_features)
        input = Variable(GI.get_node_features(Gs, nodefeat), requires_grad=True)
        
        print(input)
        
        
        
        wnet = nn.Linear(2,3*2)
        wnet.weight.data.copy_(torch.Tensor([[0,0],[1,0],[0,1],[1,1],[0,0],[2,2]]))
        print(wnet.weight)
        wnet.bias.data.fill_(1)
        
        gconv = GraphConvModule(3, 2, wnet, GI)
    
        output = gconv(input)
        
        print(output)
        
        loss = output.sum()
        loss.backward()
        
        print(input.grad)
        
        
    # todo: test with clustered edges: regular grid convolution
    
    def test_gc_identity(self):
    
        def my_edge_features(edges):
            feat = [ (e['d'] if 'd' in e else torch.Tensor([[0, 0]])) for e in edges ]
            feat = torch.cat(feat, 0)
            return feat, None    
    
        nodefeat = torch.Tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
    
        G = nx.DiGraph()
        G.add_nodes_from([0,1,2,3,4])
        G.add_edges_from([(0,0,{'d':torch.Tensor([[1]])}), 
                          (1,1,{'d':torch.Tensor([[1]])}),
                          (1,4,{'d':torch.Tensor([[0]])}),
                          (2,2,{'d':torch.Tensor([[1]])}),
                          (3,3,{'d':torch.Tensor([[1]])}),
                          (4,4,{'d':torch.Tensor([[1]])})])
                          
        GI = GraphConvInfo()
        GI.set_batch(G, my_edge_features)
        input = Variable(GI.get_node_features(G, nodefeat), requires_grad=True)
        
        gconv = GraphConvModule(3, 3, None, GI)
        gconv._mean = False
        output = gconv(input)
        
        print(input, output)
        self.assertTrue(torch.norm(input-output)==0)
    
    
    def xtest_get_degrees(self):       
        idxn = torch.LongTensor([0,1,2,1,2])
        degs = [3,2]
        weights = Variable(torch.Tensor([[1], [2], [3], [4], [5]]))
        wadj = torch.Tensor([[1, 0],
                             [2, 4],
                             [3, 5]])
        
        gdf = GetDegreesFunction(idxn, None, degs, 3)
    
        s, d, w = gdf(weights)

        print(s, d, wadj.sum(1), wadj.sum(0))
       
    def xtest_get_degrees_gradcheck(self):

        torch.set_default_tensor_type('torch.DoubleTensor') #necessary for proper numerical gradient
 
        # without idxe
        n,e,in_channels = 20,50,10
        weights = torch.randn(e,in_channels)
        idxn = torch.from_numpy(np.random.randint(n,size=e))
        idxe = None
        degs = [5, 0, 15, 20, 10]  #strided conv

        func = GetDegreesFunction(idxn, idxe, degs, n)
        data = [Variable(weights, requires_grad=True)]
        ok = gradcheck(func, data)
        self.assertTrue(ok)
        
        # with idxe
        weights = torch.randn(30,in_channels)
        idxe = torch.from_numpy(np.random.randint(30,size=e))

        func = GetDegreesFunction(idxn, idxe, degs, n)
        ok = gradcheck(func, data)
        self.assertTrue(ok)
        
        torch.set_default_tensor_type('torch.FloatTensor')
    
    
        
if __name__ == '__main__':
    unittest.main()        