import torch
import torch.nn as nn
import torchnet as tnt
import time
from torch.autograd import Variable, Function
from .GraphConvInfo import GraphConvInfo


def appending_unsqueeze_as(x,t):
    for i in range(t.dim()-x.dim()):
        x = x.unsqueeze(-1)
    return x

#intervals = [tnt.meter.AverageValueMeter(), tnt.meter.AverageValueMeter(), tnt.meter.AverageValueMeter()]

class GraphConvFunction(Function):
    AGGR_MEAN = 0
    AGGR_SUM = 1
    AGGR_MAX = 2

    def __init__(self, in_channels, out_channels, idxn, idxe, degs, edges_degnorm=None, edge_mem_limit=1e20, aggr=0):#=GraphConvFunction.AGGR_MEAN):
        super(Function, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._idxn = idxn
        self._idxe = idxe
        self._degs = degs
        self._edges_degnorm = edges_degnorm
        self._edge_mem_limit = edge_mem_limit
        self._aggr = aggr
        
    def _multiply(self, a, b, out, f_a=None, f_b=None):
        """ """
        if self._full_weight_mat:
            # weights are full in_channels x out_channels matrices -> mm
            torch.bmm(f_a(a) if f_a else a, f_b(b) if f_b else b, out=out)        
        else:
            # weights represent diagonal matrices or weighted identity matrices -> cmul
            torch.mul(a, b.expand_as(a), out=out)
                
    def forward(self, input, weights):
        self.save_for_backward(input, weights)
        
        if self._edges_degnorm is not None: 
            self._edges_degnorm = appending_unsqueeze_as(self._edges_degnorm, weights)        
        self._full_weight_mat = weights.dim()==3
        assert self._full_weight_mat or (self._in_channels==self._out_channels and weights.size(1) in [1, self._in_channels])

        output = input.new(len(self._degs), self._out_channels)
        if self._aggr==GraphConvFunction.AGGR_MAX:
            self._max_indices = self._idxn.new(len(self._degs), self._out_channels).fill_(13)
        
        sel_input, sel_weights, products = input.new(), input.new(), input.new()
        
        startd, starte = 0, 0
        while startd < len(self._degs):
            numd, nume = 0, 0
            for d in range(startd, len(self._degs)):
                if nume==0 or nume + self._degs[d] <= self._edge_mem_limit:
                    nume += self._degs[d]
                    numd += 1
                else:
                    break
            
            #torch.cuda.synchronize()
            #t = time.time()
            
            torch.index_select(input, 0, self._idxn.narrow(0,starte,nume), out=sel_input)
            
            if self._idxe is not None:
                torch.index_select(weights, 0, self._idxe.narrow(0,starte,nume), out=sel_weights)
                if self._edges_degnorm is not None:
                    sel_weights.mul_(self._edges_degnorm.narrow(0,starte,nume).expand_as(sel_weights))
            else:
                if self._edges_degnorm is not None:
                    torch.mul(weights.narrow(0,starte,nume), self._edges_degnorm.narrow(0,starte,nume).expand_as(weights.narrow(0,starte,nume)), out=sel_weights)
                else:
                    sel_weights = weights.narrow(0,starte,nume)
                
            #torch.cuda.synchronize()
            #intervals[0].add(1000*(time.time()-t))
            #t = time.time()               

            self._multiply(sel_input, sel_weights, products, lambda a: a.unsqueeze(1))

            #torch.cuda.synchronize()
            #intervals[1].add(1000*(time.time()-t))
            #t = time.time()       
            
            k = 0
            for i in range(startd, startd+numd):
                if self._degs[i]>0:
                    if self._aggr==GraphConvFunction.AGGR_MEAN:
                        torch.mean(products.narrow(0,k,self._degs[i]), 0, out=output[i])
                    elif self._aggr==GraphConvFunction.AGGR_MAX:
                        torch.max(products.narrow(0,k,self._degs[i]), 0, out=(output[i], self._max_indices[i]))
                    elif self._aggr==GraphConvFunction.AGGR_SUM:
                        torch.sum(products.narrow(0,k,self._degs[i]), 0, out=output[i])
                else:
                    output[i].fill_(0)
                k = k + self._degs[i]
 
            #torch.cuda.synchronize()
            #intervals[2].add(1000*(time.time()-t))
                    
            startd += numd
            starte += nume  

        #print(intervals[0].value(), intervals[1].value(), intervals[2].value())    
    
        return output



    def backward(self, grad_output):
        input, weights = self.saved_tensors

        grad_input = input.new(input.size()).fill_(0)
        grad_weights = weights.new(weights.size())
        if self._idxe is not None: grad_weights.fill_(0)
        
        grad_products, sel_input, tmp = input.new(), input.new(), input.new()

        startd, starte = 0, 0
        while startd < len(self._degs):
            numd, nume = 0, 0
            for d in range(startd, len(self._degs)):
                if nume==0 or nume + self._degs[d] <= self._edge_mem_limit:
                    nume += self._degs[d]
                    numd += 1
                else:
                    break                    
            
            grad_products.resize_(nume, self._out_channels)

            k = 0
            for i in range(startd, startd+numd):
                if self._degs[i]>0:
                    if self._aggr==GraphConvFunction.AGGR_MEAN:
                        torch.div(grad_output[i], self._degs[i], out=grad_products[k])
                        if self._degs[i]>1:
                            grad_products.narrow(0, k+1, self._degs[i]-1).copy_( grad_products[k].expand(self._degs[i]-1,1,self._out_channels) )
                    elif self._aggr==GraphConvFunction.AGGR_MAX:
                        grad_products.narrow(0, k, self._degs[i]).fill_(0).scatter_(0, self._max_indices[i].view(1,-1), grad_output[i].view(1,-1))  
                    elif self._aggr==GraphConvFunction.AGGR_SUM:
                        grad_products.narrow(0, k, self._degs[i]).copy_( grad_output[i].expand(self._degs[i],1,self._out_channels) )
                    k = k + self._degs[i]    

            # grad wrt weights
            torch.index_select(input, 0, self._idxn.narrow(0,starte,nume), out=sel_input)
            
            if self._idxe is not None:
                self._multiply(sel_input, grad_products, tmp, lambda a: a.unsqueeze(1).transpose_(2,1), lambda b: b.unsqueeze(1))                
                if self._edges_degnorm is not None:
                    tmp.mul_(self._edges_degnorm.narrow(0,starte,nume).expand_as(tmp))
                grad_weights.index_add_(0, self._idxe.narrow(0,starte,nume), tmp)
            else:
                self._multiply(sel_input, grad_products, grad_weights.narrow(0,starte,nume), lambda a: a.unsqueeze(1).transpose_(2,1), lambda b: b.unsqueeze(1))
                if self._edges_degnorm is not None:
                    grad_weights.narrow(0,starte,nume).mul_(self._edges_degnorm.narrow(0,starte,nume).expand_as(grad_weights.narrow(0,starte,nume)))                
                    
            # grad wrt input
            if self._idxe is not None:
                torch.index_select(weights, 0, self._idxe.narrow(0,starte,nume), out=tmp)
                if self._edges_degnorm is not None:
                    tmp.mul_(self._edges_degnorm.narrow(0,starte,nume).expand_as(tmp))                
                self._multiply(grad_products, tmp, sel_input, lambda a: a.unsqueeze(1), lambda b: b.transpose_(2,1))
            else:
                if self._edges_degnorm is not None:
                    torch.mul(weights.narrow(0,starte,nume), self._edges_degnorm.narrow(0,starte,nume).expand_as(weights.narrow(0,starte,nume)), out=tmp)
                    self._multiply(grad_products, tmp, sel_input, lambda a: a.unsqueeze(1), lambda b: b.transpose_(2,1))                    
                else:
                    self._multiply(grad_products, weights.narrow(0,starte,nume), sel_input, lambda a: a.unsqueeze(1), lambda b: b.transpose_(2,1))                     

            grad_input.index_add_(0, self._idxn.narrow(0,starte,nume), sel_input)
                    
            startd += numd
            starte += nume   
       
        return grad_input, grad_weights



        
        
        
        
        
        
        
        
        
        
class GetDegreesFunction(Function):
    """ Given weights, extracts out-degrees and in-degrees of the respective weighted graph (corresponds to row and col sums of weighted adjacency mat). 
    Returns them in edge-expanded form (degrees for each edge). Weights can be multidimensional. Returns edge-expanded weights too, for convenience."""

    def __init__(self, idxn, idxe, degs, nnodes):
        super(Function, self).__init__()
        self._idxn = idxn
        self._idxe = idxe
        self._degs = degs
        self._nnodes = nnodes
                       
    def forward(self, weights):       
        assert weights.dim()==2
        self.weight_size = weights.size()

        sel_weights = weights if self._idxe is None else torch.index_select(weights, 0, self._idxe)
        
        # dest nodes / in-degrees: just sum segments of weights
        dest_degs_e = weights.new(sel_weights.size())
        k = 0
        for i in range(len(self._degs)):
            if self._degs[i]>0:
                torch.sum(sel_weights.narrow(0,k,self._degs[i]), 0, out=dest_degs_e[k])
                if self._degs[i]>1:
                    dest_degs_e.narrow(0, k+1, self._degs[i]-1).copy_( dest_degs_e[k].expand(self._degs[i]-1,1,self.weight_size[1]) )
            k = k + self._degs[i]
        
        # src nodes / out-degrees: sum-and-distribute via indices
        src_degs = weights.new(self._nnodes, self.weight_size[1]).fill_(0)
        src_degs.index_add_(0, self._idxn, sel_weights)
        src_degs_e = torch.index_select(src_degs, 0, self._idxn)
        
        return src_degs_e, dest_degs_e, sel_weights
      
    def backward(self, grad_src_degs_e, grad_dest_degs_e, grad_sel_weights):
        # dest nodes / in-degrees
        grad_weights_d = grad_src_degs_e.new(self._idxn.size(0), self.weight_size[1])
        k = 0
        for i in range(len(self._degs)):
            if self._degs[i]>0:
                torch.sum(grad_dest_degs_e.narrow(0,k,self._degs[i]), 0, out=grad_weights_d[k])
                if self._degs[i]>1:
                    grad_weights_d.narrow(0, k+1, self._degs[i]-1).copy_( grad_weights_d[k].expand(self._degs[i]-1,1,self.weight_size[1]) )
            k = k + self._degs[i]    

        # src nodes / out-degrees
        grad_src_degs = grad_src_degs_e.new(self._nnodes, self.weight_size[1]).fill_(0)
        grad_src_degs.index_add_(0, self._idxn, grad_src_degs_e)    
        grad_weights_s = torch.index_select(grad_src_degs, 0, self._idxn) 
        
        grad_weights_d.add_(grad_weights_s).add_(grad_sel_weights)
        if self._idxe is not None:
            grad_weights = grad_src_degs_e.new(self.weight_size).fill_(0)
            grad_weights.index_add_(0, self._idxe, grad_weights_d)
        else:
            grad_weights = grad_weights_d
        return grad_weights
        
 
class NormalizationByStationaryDist(Function): #unsuccessful attempt to normalizing by directed laplacian (its conditions not fulfilled)

    def __init__(self, idxn, idxd, idxe, degs):
        super(Function, self).__init__()
        self._idxn = idxn.cpu().numpy()    
        self._idxd = idxd.cpu().numpy()    
        self._idxe = idxe.cpu()
        self._degs = degs
                
    def forward(self, weights_):
        import scipy.sparse as sp
        import numpy as np
        
        weights = weights_.cpu()
        weights = weights.abs()
        assert weights.dim()==2 and weights.lt(0).sum()==0, 'only pos elems'
        allweightsnp = torch.index_select(weights, 0, self._idxe).numpy()       
        normalizations = []

        for d in range(allweightsnp.shape[1]):
            adjmat = sp.coo_matrix((allweightsnp[:,d], (self._idxn, self._idxd)))
            d_inv = np.reciprocal(np.array(adjmat.sum(1))) #1/outdeg            
            transmat = adjmat.dot(sp.diags(d_inv.flatten()))

            # compute vector of stationary probabilities (following Boley: Commute times for a directed graph using an asymmetric Laplacian, p 225)
            v,w = sp.linalg.eigs(transmat.transpose(), k=1) #transposed, as we want right eigenvector
            assert abs(v[0]-1.0)<1e-5
            stationaryvec = w.flatten() / w.sum()
            assert (stationaryvec.real >= 0).all() and stationaryvec.imag.sum() == 0 #hmm, often happens with ReLU (some edges are 0, ie nonexistent, and graph not strongly conn?)
            stationaryvec = stationaryvec.real
            assert(np.linalg.norm(stationaryvec.transpose() * transmat - stationaryvec) < 1e-5)
            
            # hacky normalization weight inspired by their diagonally scaled Laplacian (p 229)
            sq = np.sqrt(stationaryvec)
            print(self._idxn.dtype, self._idxd.dtype)
            normalizations.append( np.reciprocal( sq[self._idxn] * transmat.data[self._idxn] * sq[self._idxd] ) )

        normalizations = np.vstack(normalizations).transpose()
        normalizations = torch.from_numpy(normalizations)
        return normalizations.cuda() if weights_.is_cuda else normalizations
        
    def backward(self, grad_output):
        return None #can't backprop through eigs
   





class GraphConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, filter_net, gc_info=None, degree_normalization='', nrepeats=1, edge_mem_limit=1e20, aggr=''):
        super(GraphConvModule, self).__init__()
        
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._fnet = filter_net
        self._degree_normalization = degree_normalization
        self._nrepeats = nrepeats
        self._edge_mem_limit = edge_mem_limit
        self._aggr = GraphConvFunction.AGGR_MEAN if self._degree_normalization=='' else GraphConvFunction.AGGR_SUM
        if aggr=='sum': self._aggr = GraphConvFunction.AGGR_SUM
        if aggr=='max': self._aggr = GraphConvFunction.AGGR_MAX
        print(self._aggr)
        
        self.set_info(gc_info)
        
    def set_info(self, gc_info):
        self._gci = gc_info
        
    def forward(self, input):       
        idxn, idxe, degs, edgefeats = self._gci.get_buffers(input.is_cuda)
        edges_degnorm = self._gci.get_degnorm(input.is_cuda) if self._degree_normalization=='unweighted' else None
        edgefeats = Variable(edgefeats, requires_grad=False)
        
        weights = self._fnet(edgefeats) if self._fnet else edgefeats
        if weights.size(1) == self._in_channels*self._out_channels:
            weights = weights.view(-1, self._in_channels, self._out_channels)
        assert input.dim()==2 and weights.dim()==3
           
        #if self._degree_normalization=='weighted_directed':
        #   idxn_cpu, idxe_cpu, _, _ = self._gci.get_buffers(False)
        #   idxd_cpu = self._gci.get_idxd(False)            
        #   edges_degnorm = NormalizationByStationaryDist(idxn_cpu, idxd_cpu, idxe_cpu, degs)(weights).data

        if self._degree_normalization=='weighted':
            assert (weights.data >= 0).all()
            odeg, ideg, sel_weights = GetDegreesFunction(idxn, idxe, degs, input.size(0))(weights) 
            normalization = torch.sqrt(torch.mul(odeg, ideg)) # ~symmetric Laplacian
            weights = torch.div(sel_weights, normalization.add_(1e-8)) #not mem-efficient, but easy to customize what to do with odeg, ideg :)
            idxe = None
        
        for r in range(self._nrepeats):
            input = GraphConvFunction(self._in_channels, self._out_channels, idxn, idxe, degs, edges_degnorm, self._edge_mem_limit, self._aggr)(input, weights)
        return input
        
        
    def forwardxxxxxx(self, input): #rewriting it in loops won't help because everything is staying in the computation graph anyway  (would be also problematic for tensorflow?)

        idxn, idxe, degs, edgefeats = self._gci.get_buffers(input.is_cuda)
        idxn = Variable(idxn, requires_grad=False)
        if idxe: idxe = Variable(idxe, requires_grad=False)
        edgefeats = Variable(edgefeats, requires_grad=False)
           
        weights = self._fnet(edgefeats).view(-1, self._in_channels, self._out_channels)
        
        output = Variable(input.data.new(len(degs), self._out_channels))
        #output_list = []
        
        startd, starte = 0, 0
        while startd < len(degs):
            numd, nume = 0, 0
            for d in range(startd, len(degs)):
                if nume==0 or nume + degs[d] <= self.edge_mem_limit:
                    nume += degs[d]
                    numd += 1
                else:
                    break

            sel_input = torch.index_select(input, 0, idxn.narrow(0,starte,nume))
            
            if idxe:
                sel_weights = torch.index_select(weights, 0, idxe.narrow(0,starte,nume))
            else:
                sel_weights = weights.narrow(0,starte,nume)

            products = torch.bmm(sel_input.view(-1,1,self._in_channels), sel_weights)
            
            k = 0
            for i in range(startd, startd+numd):
                if degs[i]>0:
                    output.index_copy_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), torch.mean(products.narrow(0,k,degs[i]), 0).view(1,-1))
                    #output_list.append(torch.mean(products.narrow(0,k,degs[i]), 0).view(1,-1))
                else:
                    output.index_fill_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), 0)
                k = k + degs[i]   
                    
            startd += numd
            starte += nume
            
            
            #from   ecc.visualize import make_dot
            #dot = make_dot(output)
            #dot.render('/home/simonovm/workspace/imperial/population-gcn/ecc/dot' + str(startd))

        #output = torch.cat(output_list)    
            
        return output        
        
    def forwardxxx(self, input):
        assert self._degree_normalization=='', "todo"
        idxn, idxe, degs, edgefeats = self._gci.get_buffers(input.is_cuda)
        idxn = Variable(idxn, requires_grad=False)
        edgefeats = Variable(edgefeats, requires_grad=False)
    
        weights = self._fnet(edgefeats)
        if weights.size(1) == self._in_channels*self._out_channels:
            weights = weights.view(-1, self._in_channels, self._out_channels)
            
        if idxe is not None:
            idxe = Variable(idxe, requires_grad=False)
            weights = torch.index_select(weights, 0, idxe)        
        
        sel_input = torch.index_select(input, 0, idxn)

        if weights.dim()==3:
            products = torch.bmm(sel_input.view(-1,1,self._in_channels), weights)
        else:
            products = torch.mul(sel_input, weights.expand_as(sel_input))
        
        output = Variable(input.data.new(len(degs), self._out_channels))
        
        k = 0
        for i in range(len(degs)):
            if degs[i]>0:
                output.index_copy_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), torch.mean(products.narrow(0,k,degs[i]), 0).view(1,-1))
            else:
                output.index_fill_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), 0)
            k = k + degs[i]

        return output
    
    
     
class GraphTripletConvModule(nn.Module):
    """ Form of neighborhood info aggregation where the signal coming over an edge is h(g(node1),g(node2),f(edge)) instead of f(edge)*node2, thus dropping the explicit multiplication in favor of a generalized net. This allows the network to consider all participants but it's less likely that it will behave as a convolution (node2's signal will be very disturbed)."""
    def __init__(self, edgenet, nodenet, mixnet, gc_info=None, aggr='mean'):
        super(GraphTripletConvModule, self).__init__()
        self._edgenet = edgenet
        self._nodenet = nodenet
        self._mixnet = mixnet
        assert aggr in ['sum', 'mean']
        self._aggr_sum = aggr=='sum'
        
        self.set_info(gc_info)
        
    def set_info(self, gc_info):
        self._gci = gc_info          
        
    def forward(self, input):
        idxn, idxe, degs, edgefeats = self._gci.get_buffers(input.is_cuda)
        idxd = self._gci.get_idxd(input.is_cuda)
        
        idxn = Variable(idxn, requires_grad=False)
        idxd = Variable(idxd, requires_grad=False)
        edgefeats = Variable(edgefeats, requires_grad=False)        

        edgefeats = self._edgenet(edgefeats) if self._edgenet else edgefeats
        nodefeats = self._nodenet(input) if self._nodenet else input

        node1feats = torch.index_select(nodefeats, 0, idxn)
        node2feats = torch.index_select(nodefeats, 0, idxd)
        if idxe is not None:
            idxe = Variable(idxe, requires_grad=False)
            edgefeats = torch.index_select(edgefeats, 0, idxe)          
        
        feat = torch.cat([node1feats, node2feats, edgefeats], 1)
        feat = self._mixnet(feat)
        
        output = Variable(input.data.new(len(degs), feat.size(1)))
        
        k = 0
        for i in range(len(degs)):
            if degs[i]>0:
                if self._aggr_sum: 
                    output.index_copy_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), torch.sum(feat.narrow(0,k,degs[i]), 0).view(1,-1))
                else:
                    output.index_copy_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), torch.mean(feat.narrow(0,k,degs[i]), 0).view(1,-1))
            else:
                output.index_fill_(0, Variable(torch.Tensor([i]).type_as(idxn.data)), 0)
            k = k + degs[i]

        return output    