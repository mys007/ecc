import torch
import torch.nn as nn
import torchnet as tnt
import time
from torch.autograd import Variable, Function
from .GraphPoolInfo import GraphPoolInfo


#intervals = [tnt.meter.AverageValueMeter(), tnt.meter.AverageValueMeter(), tnt.meter.AverageValueMeter()]

class GraphPoolFunction(Function):
    AGGR_MEAN = 0
    AGGR_MAX = 1

    def __init__(self, idxn, degs, aggr, edge_mem_limit=1e20):
        super(Function, self).__init__()
        self._idxn = idxn
        self._degs = degs
        self._edge_mem_limit = edge_mem_limit
        self._aggr = aggr

                
    def forward(self, input):
        self.save_for_backward(input)
        
        output = input.new(len(self._degs), input.size(1))
        if self._aggr==GraphConvFunction.AGGR_MAX:
            self._max_indices = self._idxn.new(len(self._degs), input.size(1)).fill_(1e10)
        
        sel_input = input.new()
        
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
            
            k = 0
            for i in range(startd, startd+numd):
                if self._degs[i]>0:
                    if self._aggr==GraphConvFunction.AGGR_MEAN:
                        torch.mean(sel_input.narrow(0,k,self._degs[i]), 0, out=output[i])
                    elif self._aggr==GraphConvFunction.AGGR_MAX:
                        torch.max(sel_input.narrow(0,k,self._degs[i]), 0, out=(output[i], self._max_indices[i]))
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
        input = self.saved_tensors

        grad_input = input.new(input.size()).fill_(0)
        
        grad_sel_input = input.new()

        startd, starte = 0, 0
        while startd < len(self._degs):
            numd, nume = 0, 0
            for d in range(startd, len(self._degs)):
                if nume==0 or nume + self._degs[d] <= self._edge_mem_limit:
                    nume += self._degs[d]
                    numd += 1
                else:
                    break                    
            
            grad_sel_input.resize_(nume, input.size(1))

            k = 0
            for i in range(startd, startd+numd):
                if self._degs[i]>0:
                    if self._aggr==GraphConvFunction.AGGR_MEAN:
                        torch.div(grad_output[i], self._degs[i], out=grad_sel_input[k])
                        if self._degs[i]>1:
                            grad_sel_input.narrow(0, k+1, self._degs[i]-1).copy_( grad_sel_input[k].expand(self._degs[i]-1,1,self._out_channels) )
                    elif self._aggr==GraphConvFunction.AGGR_MAX:
                        grad_sel_input.narrow(0, k, self._degs[i]).fill_(0).scatter_(0, self._max_indices[i].view(1,-1), grad_output[i].view(1,-1))
                    k = k + self._degs[i]             

            grad_input.index_add_(0, self._idxn.narrow(0,starte,nume), grad_sel_input)
                    
            startd += numd
            starte += nume   
       
        return grad_input
        
        
        
class GraphPoolModule(nn.Module):
    def __init__(self, aggr, gc_info=None, edge_mem_limit=1e20):
        super(GraphPoolModule, self).__init__()
        
        self._aggr = aggr
        self._edge_mem_limit = edge_mem_limit       
        self.set_info(gc_info)
        
    def set_info(self, gp_info):
        self._gpi = gp_info
        
    def forward(self, input):       
        idxn, degs = self._gpi.get_buffers(input.is_cuda)
        return GraphPoolFunction(idxn, degs, self._aggr, self._edge_mem_limit)(input)
        
        
class GraphAvgPoolModule(GraphPoolModule):
    def __init__(self, gc_info=None, edge_mem_limit=1e20):
        super(GraphAvgPoolModule, self).__init__(GraphPoolFunction.AGGR_MEAN, gc_info, edge_mem_limit)        
        
class GraphMaxPoolModule(GraphPoolModule):
    def __init__(self, gc_info=None, edge_mem_limit=1e20):
        super(GraphAvgPoolModule, self).__init__(GraphPoolFunction.AGGR_MAX, gc_info, edge_mem_limit)                