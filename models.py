import numpy as np
import torch
import torch.nn as nn
import ecc
    
def create_fnet(nchannels, nfeat, nfeato, orthoinit, llbias):
    fnet_modules = []   
    for k in range(len(nchannels)-1):
        fnet_modules.append(nn.Linear(nchannels[k], nchannels[k+1]))
        if orthoinit: nn.init.orthogonal(fnet_modules[-1].weight, gain=nn.init.calculate_gain('relu'))
        fnet_modules.append(nn.ReLU(True))                    
    fnet_modules.append(nn.Linear(nchannels[-1], nfeat*nfeato, bias=llbias))
    if orthoinit: nn.init.orthogonal(fnet_modules[-1].weight)
    return nn.Sequential(*fnet_modules)
            

class CloudNetwork(nn.Module):
    def __init__(self, config, nfeat, ch_fnet, orthoinit=True, fnet_llbias=True):
        # ch_fnet: without the last layer
    
        super(CloudNetwork, self).__init__()

        self.gconvs = []
        self.gpools = []
        self.pyramid_conf = []
        
        for d, conf in enumerate(config.split(',')):
            conf = conf.split('_')
            
            if conf[0]=='f':    #args: output_feats
                self.add_module(str(d)+':fc', nn.Linear(nfeat, int(conf[1])))
                nfeat = int(conf[1])
            elif conf[0]=='b':  #args: not_affine
                self.add_module(str(d)+':b', nn.BatchNorm1d(nfeat, eps=1e-5, affine=len(conf)==1))
            elif conf[0]=='r':    
                self.add_module(str(d)+':r', nn.ReLU(True))
            elif conf[0]=='d': #args: dropout_prob    
                self.add_module(str(d)+':d', nn.Dropout(p=float(conf[1]), inplace=False))   

            elif conf[0]=='m' or conf[0]=='a': #args: output_resolution, output_radius   
                res, rad = float(conf[1]), float(conf[2])
                assert pyramid_conf[-1][0] < res, "Pooling should coarsen resolution."
                self.pyramid_conf.append((res,rad))

                gpool = ecc.GraphMaxPoolModule() if conf[0]=='m' else ecc.GraphAvgPoolModule()
                self.gpools.append(gpool)
                self.add_module(str(d)+':p', gpool)   
                
            elif conf[0]=='i': #args: initial_resolution, initial_radius
                res, rad = float(conf[1]), float(conf[2])
                assert len(pyramid_conf)==0 or pyramid_conf[-1][0]==res, "Graph cannot be coarsened directly"
                self.pyramid_conf.append((res,rad))           
                
            elif conf[0]=='c': #args: output_feats
                nfeato = int(conf[1])
                assert len(self.pyramid_conf)>0, "Convolution needs defined graph"

                fnet = create_fnet(ch_fnet, nfeat, nfeato, orthoinit, fnet_llbias)
                gconv = ecc.GraphConvModule(nfeat, nfeato, fnet)
                self.gconvs.append((gconv, len(self.pyramid_conf)-1))
                self.add_module(str(d)+':gc', gconv)     
                nfeat = nfeato
      
            else:
                raise NotImplementedError
                
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
        
        