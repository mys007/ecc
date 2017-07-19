import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import ecc
    
def create_fnet(nchannels, nfeat, nfeato, orthoinit, llbias):
    fnet_modules = []   
    for k in range(len(nchannels)-1):
        fnet_modules.append(nn.Linear(nchannels[k], nchannels[k+1]))
        if orthoinit: init.orthogonal(fnet_modules[-1].weight, gain=init.calculate_gain('relu'))
        fnet_modules.append(nn.ReLU(True))                    
    fnet_modules.append(nn.Linear(nchannels[-1], nfeat*nfeato, bias=llbias))
    if orthoinit: init.orthogonal(fnet_modules[-1].weight)
    return nn.Sequential(*fnet_modules)
            

class CloudNetwork(nn.Module):
    def __init__(self, config, nfeat, fnet_widths, fnet_orthoinit=True, fnet_llbias=True, edge_mem_limit=1e20):
    
        super(CloudNetwork, self).__init__()

        self.gconvs = []
        self.gpools = []
        self.pyramid_conf = []
        
        for d, conf in enumerate(config.split(',')):
            conf = conf.strip().split('_')
            
            if conf[0]=='f':    #args: output_feats
                self.add_module(str(d), nn.Linear(nfeat, int(conf[1])))
                nfeat = int(conf[1])
            elif conf[0]=='b':  #args: not_affine
                self.add_module(str(d), nn.BatchNorm1d(nfeat, eps=1e-5, affine=len(conf)==1))
            elif conf[0]=='r':    
                self.add_module(str(d), nn.ReLU(True))
            elif conf[0]=='d': #args: dropout_prob    
                self.add_module(str(d), nn.Dropout(p=float(conf[1]), inplace=False))   

            elif conf[0]=='m' or conf[0]=='a': #args: output_resolution, output_radius   
                res, rad = float(conf[1]), float(conf[2])
                assert self.pyramid_conf[-1][0] < res, "Pooling should coarsen resolution."
                self.pyramid_conf.append((res,rad))

                gpool = ecc.GraphMaxPoolModule() if conf[0]=='m' else ecc.GraphAvgPoolModule()
                self.gpools.append(gpool)
                self.add_module(str(d), gpool)   
                
            elif conf[0]=='i': #args: initial_resolution, initial_radius
                res, rad = float(conf[1]), float(conf[2])
                assert len(self.pyramid_conf)==0 or self.pyramid_conf[-1][0]==res, "Graph cannot be coarsened directly"
                self.pyramid_conf.append((res,rad))           
                
            elif conf[0]=='c': #args: output_feats
                nfeato = int(conf[1])
                assert len(self.pyramid_conf)>0, "Convolution needs defined graph"

                fnet = create_fnet(fnet_widths, nfeat, nfeato, fnet_orthoinit, fnet_llbias)
                gconv = ecc.GraphConvModule(nfeat, nfeato, fnet, edge_mem_limit=edge_mem_limit)
                self.gconvs.append((gconv, len(self.pyramid_conf)-1))
                self.add_module(str(d), gconv)     
                nfeat = nfeato
      
            else:
                raise NotImplementedError('Unknown module: ' + conf[0])
                
    def set_info(self, gc_infos, gp_infos):
        gc_infos = gc_infos if isinstance(gc_infos,(list,tuple)) else [gc_infos]
        gp_infos = gp_infos if isinstance(gp_infos,(list,tuple)) else [gp_infos]

        for gc,i in self.gconvs:
            gc.set_info(gc_infos[i])
        for i,gp in enumerate(self.gpools):
            gp.set_info(gp_infos[i])
            
        
    def forward(self, input):        
        for module in self._modules.values():
            input = module(input)
        return input
        
        