from __future__ import division
from __future__ import print_function
from builtins import range

import time
import random
import numpy as np
import json
import os
import sys
import math
import argparse   
import ast

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
import networkx as nx
import functools

import pointcloud_dataset
import models
import ecc
    
def main():
    parser = argparse.ArgumentParser(description='ECC')  
    
    
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--lr', default=0.005, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--lr_steps', default='[]')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=32, type=int, help='Minibatch size')
    parser.add_argument('--batch_parts', default=1, type=int, help='Minibatch can be evaluated in multiple shards, >=1 (useful to save memory, though computation not equivalent due to batchnorms)')
    
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation (default: 123)')    
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--nworkers', default=0, type=int, help='num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    
    parser.add_argument('--dataset', default='sydney')    
    parser.add_argument('--cvfold', default=0, type=int, help='Fold left-out for testing in leave-one-out setting (Sydney,...)')
    parser.add_argument('--odir', default='results')   
    
    parser.add_argument('--model_config', default='i_0.1_0.2,c_16,b,r,d_0.3,m_1e10_1e10,b,r,f_14')  
    
    parser.add_argument('--pc_augm_input_dropout', default=0.1, type=float, help='Training augmentation: Probability of removing points in input point clouds')
    parser.add_argument('--pc_augm_scale', default=1.1, type=float, help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=1, type=int, help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0.5, type=float, help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_knn', default=0, type=int, help='Bool, use knn-search instead of radius-search for neighborhood building')
    parser.add_argument('--pc_attribs', default='polareukl', help='Edge attribute definition: eukl|polar coordinates')
    parser.add_argument('--edgecompaction', default=1, type=int, help='Bool, compact all edges attribute vectors into a unique set, evaluate and redistribute the results (faster and more mem-efficient if edge attibutes tend to repeat, usually with categorical types).')
    
    parser.add_argument('--fnet_widths', default='[]', help='List of width of hidden filter gen net layers (excluding the input and output ones, they are automatic)')
    parser.add_argument('--fnet_llbias', default=1, type=int, help='Bool, use bias in the last layer in filter gen net')
    parser.add_argument('--fnet_orthoinit', default=1, type=int, help='Bool, use orthogonal weight initialization for filter gen net.')
    
    
    #--lr 0.1 --lr_steps '[200,245]' --epochs 250  --batch_size 32 --model_config 'i_0.1_0.2, c_16,b,r, c_32,b,r, m_0.25_0.5, c_32,b,r, c_32,b,r, m_0.75_1.5, c_64,b,r, m_1.5_1.5,a_1e10_1e10, f_64,b,r,d_0.2,f_14' --fnet_widths '[16,32]' --pc_augm_scale 1.2 --pc_augm_mirror_prob 0.2 --pc_augm_input_dropout 0.1 --nworkers 3 --edgecompaction 0 --fnet_llbias 0

     
    args = parser.parse_args()
    args.lr_steps = ast.literal_eval(args.lr_steps)
    args.fnet_widths = ast.literal_eval(args.fnet_widths)
    
    print('Will save to ' + args.odir)
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    with open(os.path.join(args.odir, 'cmdline.txt'), 'w') as f:
        f.write(" ".join(sys.argv))
    
    seed(args.seed, args.cuda)
    
    if args.dataset=='sydney':
        dbinfo = pointcloud_dataset.get_sydney_info(args)
        create_dataset = pointcloud_dataset.get_sydney
        edge_feat_func = pointcloud_dataset.cloud_edge_feats
    else:
        raise NotImplementedError('Unknown dataset')
        
    model = create_model(args, dbinfo)   
    train_dataset = create_dataset(args, model.pyramid_conf, True)
    test_dataset = create_dataset(args, model.pyramid_conf, False)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)  

    collate_func = functools.partial(ecc.graph_info_collate_classification, edge_func=functools.partial(edge_feat_func, args=args))
    assert args.batch_size % args.batch_parts == 0    
   
    def train(epoch):
        model.train()
        #TODO: set seed to dataset

        loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(args.batch_size/args.batch_parts), collate_fn=collate_func, num_workers=args.nworkers, shuffle=True)

        timer_loader, timer_trainer = tnt.meter.AverageValueMeter(), tnt.meter.AverageValueMeter()
        loss_meter = tnt.meter.AverageValueMeter()
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        #cm_meter = tnt.meter.ConfusionMeter(k=dbinfo['classes'])
        
        t = time.time()
        
        for bidx, (inputs, targets, GIs, PIs) in enumerate(loader):
            
            timer_loader.add(1000*(time.time()-t))


            model.set_info(GIs, PIs)
            if args.cuda: 
                inputs, targets = inputs.cuda(), targets.cuda()
        
            inputs, targets = Variable(inputs), Variable(targets)
            t = time.time()                
            if bidx % args.batch_parts == 0:
                optimizer.zero_grad()
                
            outputs = model(inputs)
            assert outputs.size(1)==dbinfo['classes'], "Model should output {:d} dimensions.".format(dbinfo['classes'])
            
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            
            timer_trainer.add(1000*(time.time()-t))   
            loss_meter.add(loss.data[0])
            acc_meter.add(outputs.data, targets.data)
            #cm_meter.add(outputs.data, targets.data)    

            #outputs = None
            #loss = None            
        
            if bidx % args.batch_parts == args.batch_parts-1:
                if args.batch_parts>0: 
                    for p in model.parameters():
                        p.grad.data.div_(args.batch_parts)
                optimizer.step()
                

            # todo: print current values, but return average values
            print(loss_meter.value(), acc_meter.value(), timer_loader.value(), timer_trainer.value())
            

                
            t = time.time()
            
        return acc_meter.value()[0], loss_meter.value()[0]
            
            
            
    def eval(epoch):
        model.eval()        
        #TODO: set seed to dataset

        loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(args.batch_size/args.batch_parts), collate_fn=collate_func, num_workers=args.nworkers)
                
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        #cm_meter = tnt.meter.ConfusionMeter(k=dbinfo['classes'])
        
        for bidx, (inputs, targets, GIs, PIs) in enumerate(loader):
            model.set_info(GIs, PIs)
            if args.cuda: 
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = model(inputs)
        
            acc_meter.add(outputs.data, targets.data)
            #cm_meter.add(outputs.data, targets.data)  
        
        return acc_meter.value()[0]
            
            
    stats = []    
    
    for epoch in range(args.epochs):        
        acc_train, loss = train(epoch)
        
        optimizer = lr_scheduler(optimizer, epoch, args, model)
        if True: #epoch % 10 == 0:
            acc_test = eval(epoch)       
        else:
            acc_test = 0
        print('Epoch {}, Train accuracy: {}, \tloss: {}, \tTest accuracy: {}'.format(epoch, acc_train, loss, acc_test))
        stats.append({'epoch': epoch, 'acc_train': acc_train, 'loss': loss, 'acc_test': acc_test})
        
        if math.isnan(loss): break

    with open(os.path.join(args.odir, 'trainlog.txt'), 'w') as outfile:
        json.dump(stats, outfile)   
    
    
def create_model(args, dbinfo):
    model = models.CloudNetwork(args.model_config, dbinfo['feats'], [dbinfo['edge_feats']] + args.fnet_widths, args.fnet_orthoinit, args.fnet_llbias)
    print('Parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print(model)    
    if args.cuda: 
        model.cuda()
    return model    
    
def seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: 
        torch.cuda.manual_seed(seed)    
    
def lr_scheduler(optimizer, epoch, args, model):
    """ Decreases learning rate at predefined steps"""
    if epoch in args.lr_steps:
        lr = args.lr * (args.lr_decay**(args.lr_steps.index(epoch)+1))
        #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)  
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('LR is set to {}'.format(lr))            
    return optimizer
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    main()
