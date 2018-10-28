"""
    Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs
    https://github.com/mys007/ecc
    https://arxiv.org/abs/1704.02901
    2017 Martin Simonovsky
"""
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
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
import functools

import pointcloud_dataset
import models
import ecc
    
    
def main():
    parser = argparse.ArgumentParser(description='ECC')  
    
    # Optimization arguments
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--lr', default=0.005, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default='[]', help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--batch_parts', default=1, type=int, help='Batch can be evaluated sequentially in multiple shards, >=1, very useful in low memory settings, though computation is not strictly equivalent due to batch normalization runnning statistics.')
    
    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--nworkers', default=0, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')
    
    # Dataset 
    parser.add_argument('--dataset', default='sydney', help='Dataset name: sydney|modelnet10|modelnet40')    
    parser.add_argument('--cvfold', default=0, type=int, help='Fold left-out for testing in leave-one-out setting (Sydney,...)')
    parser.add_argument('--odir', default='results', help='Directory to store results') 
    parser.add_argument('--resume', default='', help='Loads a previously saved model.')
        
    # Model    
    parser.add_argument('--model_config', default='i_0.1_0.2,c_16,b,r,d_0.3,m_1e10_1e10,b,r,f_14', help='Defines the model as a sequence of layers, see models.py for definitions of respective layers and acceptable arguments.')  
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation (default: 123)')        
    parser.add_argument('--test_sample_avging', default='none', help='Test-time result aggregation over dataset-specific number of augmented sample, vote|score|none')    
    
    # Point cloud processing
    parser.add_argument('--pc_augm_input_dropout', default=0.1, type=float, help='Training augmentation: Probability of removing points in input point clouds')
    parser.add_argument('--pc_augm_scale', default=1.1, type=float, help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=1, type=int, help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0.5, type=float, help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_knn', default=0, type=int, help='Bool, use knn-search instead of radius-search for neighborhood building')
    parser.add_argument('--pc_attribs', default='polareukl', help='Edge attribute definition: eukl|polar coordinates')
    
    # Filter generating network
    parser.add_argument('--fnet_widths', default='[]', help='List of width of hidden filter gen net layers (excluding the input and output ones, they are automatic)')
    parser.add_argument('--fnet_llbias', default=1, type=int, help='Bool, use bias in the last layer in filter gen net')
    parser.add_argument('--fnet_orthoinit', default=1, type=int, help='Bool, use orthogonal weight initialization for filter gen net.')
    parser.add_argument('--edgecompaction', default=1, type=int, help='Bool, compact all edges attribute vectors into a unique set (faster and more mem-efficient if edge attibutes tend to repeat, usually with categorical types).')
    parser.add_argument('--edge_mem_limit', default=1e20, type=int, help='Number of edges to process in parallel during computation, a low number can reduce memory peaks.')
        
    args = parser.parse_args()
    args.start_epoch = 0
    args.lr_steps = ast.literal_eval(args.lr_steps)
    args.fnet_widths = ast.literal_eval(args.fnet_widths)
    

    print('Will save to ' + args.odir)
    if not os.path.exists(args.odir):
        os.makedirs(args.odir)
    with open(os.path.join(args.odir, 'cmdline.txt'), 'w') as f:
        f.write(" ".join(sys.argv))
    
    seed(args.seed, args.cuda)
    logging.getLogger().setLevel(logging.INFO)  #set to logging.DEBUG to allow for more prints
    
    # Decide on the dataset
    if args.dataset=='sydney':
        dbinfo = pointcloud_dataset.get_sydney_info(args)
        create_dataset = pointcloud_dataset.get_sydney
        edge_feat_func = pointcloud_dataset.cloud_edge_feats
    elif args.dataset=='modelnet10' or args.dataset=='modelnet40':
        dbinfo = pointcloud_dataset.get_modelnet_info(args)
        create_dataset = pointcloud_dataset.get_modelnet
        edge_feat_func = pointcloud_dataset.cloud_edge_feats
    else:
        raise NotImplementedError('Unknown dataset ' + args.dataset)
    
    # Create model and optimizer
    if args.resume != '':
        model, optimizer = resume(args, dbinfo)
    else:
        model = create_model(args, dbinfo)   
        optimizer = create_optimizer(args, model)
    
    train_dataset = create_dataset(args, model.pyramid_conf, True)
    test_dataset = create_dataset(args, model.pyramid_conf, False)

    collate_func = functools.partial(ecc.graph_info_collate_classification, edge_func=functools.partial(edge_feat_func, args=args))
    assert args.batch_size % args.batch_parts == 0    
      
      
    ############
    def train(epoch):
        model.train()
        #TODO: set seed to dataset (multiprocess)

        loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(args.batch_size/args.batch_parts), collate_fn=collate_func, num_workers=args.nworkers, shuffle=True)
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=100)

        t_loader_meter, t_trainer_meter = tnt.meter.AverageValueMeter(), tnt.meter.AverageValueMeter()
        loss_meter = tnt.meter.AverageValueMeter()
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)       
        t0 = time.time()
        
        # iterate over dataset in batches
        for bidx, (inputs, targets, GIs, PIs) in enumerate(loader):
            
            t_loader = 1000*(time.time()-t0)

            model.set_info(GIs, PIs, args.cuda)
            if args.cuda: 
                inputs, targets = inputs.cuda(), targets.cuda()

            if bidx % args.batch_parts == 0:
                optimizer.zero_grad()

            t0 = time.time()                                
            outputs = model(inputs)
            assert outputs.size(1)==dbinfo['classes'], "Model should output {:d} dimensions.".format(dbinfo['classes'])
            
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()

            t_trainer = 1000*(time.time()-t0)
            loss_meter.add(loss.item())
            acc_meter.add(outputs.detach().cpu(), targets.cpu())
        
            if bidx % args.batch_parts == args.batch_parts-1: #
                if args.batch_parts>0: 
                    for p in model.parameters():
                        p.grad.div_(args.batch_parts)
                optimizer.step()
                
            t_loader_meter.add(t_loader)
            t_trainer_meter.add(t_trainer)  
            logging.debug('Batch loss %f, Loader time %f ms, Trainer time %f ms.', loss.item(), t_loader, t_trainer)
                
            t0 = time.time()

        return acc_meter.value()[0], loss_meter.value()[0], t_loader_meter.value()[0], t_trainer_meter.value()[0]
            
    ############        
    def eval(epoch):
        model.eval()        
        #TODO: set seed to dataset (multiprocess)
        
        if args.test_sample_avging != 'none':
            loader = torch.utils.data.DataLoader(test_dataset, batch_size=dbinfo['test_set_expansion'], collate_fn=collate_func, num_workers=args.nworkers)
        else:
            loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(args.batch_size/args.batch_parts), collate_fn=collate_func, num_workers=args.nworkers)
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=100)
        
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        cm_meter = tnt.meter.ConfusionMeter(k=dbinfo['classes'])        
        
        # iterate over dataset in batches
        with torch.no_grad():
            for bidx, (inputs, targets, GIs, PIs) in enumerate(loader):
                model.set_info(GIs, PIs, args.cuda)
                if args.cuda:
                    inputs = inputs.cuda()

                outputs = model(inputs).cpu()

                if args.test_sample_avging != 'none':
                    if args.test_sample_avging == 'vote':
                        _, ii = torch.max(outputs, 1)
                        outputs.fill_(0).scatter_(1, ii, 1)
                    acc_meter.add(outputs.mean(0), targets.narrow(0,1,1))
                    cm_meter.add(outputs.mean(0), targets.narrow(0,1,1))
                else:
                    acc_meter.add(outputs, targets)
                    cm_meter.add(outputs, targets)
        
        f1 = compute_F1(cm_meter.value())
        cacc = compute_class_acc(cm_meter.value())        
        return acc_meter.value()[0], f1, cacc
        

        
    stats = []    
    
    # Training loop
    for epoch in range(args.start_epoch, args.epochs):        
        print('Epoch {}/{} ({}):'.format(epoch, args.epochs, args.odir))
        
        acc_train, loss, t_loader, t_trainer = train(epoch)
        
        optimizer = lr_scheduler(optimizer, epoch, args, model)
        
        if (epoch+1) % args.test_nth_epoch == 0 or epoch+1==args.epochs:
            acc_test, f1_test, cacc_test = eval(epoch)  
            print('-> Train accuracy: {}, \tLoss: {}, \tTest accuracy: {}'.format(acc_train, loss, acc_test))
        else:
            acc_test, f1_test, cacc_test = 0, 0, 0
            print('-> Train accuracy: {}, \tLoss: {}'.format(acc_train, loss))
        
        stats.append({'epoch': epoch, 'acc_train': acc_train, 'loss': loss, 'acc_test': acc_test, 'f1_test': f1_test, 'cacc_test': cacc_test})

        if epoch % args.save_nth_epoch == 0 or epoch==args.epochs-1:
            with open(os.path.join(args.odir, 'trainlog.txt'), 'w') as outfile:
                json.dump(stats, outfile)           
            torch.save({'epoch': epoch + 1, 'args': args, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, 
                       os.path.join(args.odir, 'model.pth.tar'))
        
        if math.isnan(loss): break
         
    if len(stats)>0:
        with open(os.path.join(args.odir, 'trainlog.txt'), 'w') as outfile:
            json.dump(stats, outfile)           

    acc_test, f1_test, cacc_test = eval(args.start_epoch)
    print('-> Final test accuracy: {}, F1 score: {}, mean class accuracy: {}'.format(acc_test, f1_test, cacc_test))    

    
                        
                        
def resume(args, dbinfo):
    """ Loads model and optimizer state from a previous checkpoint. """
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model = create_model(checkpoint['args'], dbinfo) #use original arguments, architecture can't change
    optimizer = create_optimizer(args, model)
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    args.start_epoch = checkpoint['epoch']
    return model, optimizer
    
def create_model(args, dbinfo):
    """ Creates model """
    model = models.CloudNetwork(args.model_config, dbinfo['feats'], [dbinfo['edge_feats']] + args.fnet_widths, args.fnet_orthoinit, args.fnet_llbias, args.edge_mem_limit)
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print(model)    
    if args.cuda: 
        model.cuda()
    return model 

def create_optimizer(args, model):
    return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)  
    
def seed(seed, cuda=True):
    """ Sets seeds in all frameworks"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: 
        torch.cuda.manual_seed(seed)    
    
def lr_scheduler(optimizer, epoch, args, model):
    """ Decreases learning rate at predefined steps"""
    if epoch in args.lr_steps:
        lr = args.lr * (args.lr_decay**(args.lr_steps.index(epoch)+1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('LR is set to {}'.format(lr))            
    return optimizer
    
   
def compute_F1(cm):
    """ Computes mean F1 score from confusion matrix, weighted by class support (for Sydney) """
    avgwf1 = 0
    N = cm.shape[0]
    f1 = np.zeros(N)
    for i in range(N):
        pr, re = cm[i][i] / max(1,np.sum(cm[:,i])), cm[i][i] / max(1,np.sum(cm[i]))
        f1[i] = 2*pr*re/max(1,pr+re)
        avgwf1 = avgwf1 + np.sum(cm[i]) * f1[i]
    return avgwf1 / cm.sum()        
    
def compute_class_acc(cm):
    """ Computes mean class accuracy from confusion matrix (for ModelNet) """
    re = 0
    N = cm.shape[0]
    for i in range(N):
        re = re + cm[i][i] / max(1,np.sum(cm[i]))
    return re/N*100

    
    
    
    

if __name__ == "__main__":
    main()
