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
    parser.add_argument('--nworkers', default=0, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')
    
    parser.add_argument('--dataset', default='sydney')    
    parser.add_argument('--cvfold', default=0, type=int, help='Fold left-out for testing in leave-one-out setting (Sydney,...)')
    parser.add_argument('--odir', default='results') 
    parser.add_argument('--resume', default='')
    parser.add_argument('--test_sample_avging', default='none', help='Test-time result aggregation over dataset-specific number of augmented sample, vote|score|none')
    
    parser.add_argument('--model_config', default='i_0.1_0.2,c_16,b,r,d_0.3,m_1e10_1e10,b,r,f_14')  
    
    parser.add_argument('--pc_augm_input_dropout', default=0.1, type=float, help='Training augmentation: Probability of removing points in input point clouds')
    parser.add_argument('--pc_augm_scale', default=1.1, type=float, help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=1, type=int, help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0.5, type=float, help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_knn', default=0, type=int, help='Bool, use knn-search instead of radius-search for neighborhood building')
    parser.add_argument('--pc_attribs', default='polareukl', help='Edge attribute definition: eukl|polar coordinates')
    parser.add_argument('--edgecompaction', default=1, type=int, help='Bool, compact all edges attribute vectors into a unique set, evaluate and redistribute the results (faster and more mem-efficient if edge attibutes tend to repeat, usually with categorical types).')
    parser.add_argument('--edge_mem_limit', default=1e20, type=int, help='')
    
    parser.add_argument('--fnet_widths', default='[]', help='List of width of hidden filter gen net layers (excluding the input and output ones, they are automatic)')
    parser.add_argument('--fnet_llbias', default=1, type=int, help='Bool, use bias in the last layer in filter gen net')
    parser.add_argument('--fnet_orthoinit', default=1, type=int, help='Bool, use orthogonal weight initialization for filter gen net.')
    
    
    #CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.1 --lr_steps '[200,245]' --epochs 250  --batch_size 32 --model_config 'i_0.1_0.2, c_16,b,r, c_32,b,r, m_0.25_0.5, c_32,b,r, c_32,b,r, m_0.75_1.5, c_64,b,r, m_1.5_1.5,a_1e10_1e10, f_64,b,r,d_0.2,f_14' --fnet_widths '[16,32]' --pc_augm_scale 1.2 --pc_augm_mirror_prob 0.2 --pc_augm_input_dropout 0.1 --nworkers 3 --edgecompaction 0 --fnet_llbias 0    -> 0.7760911607844838 f1  (0.8 point less than torch, that's ok)
    
    #CUDA_VISIBLE_DEVICES=1 python main.py --dataset modelnet10 --test_nth_epoch 25 --lr 0.1 --lr_steps '[50,100,150]' --epochs 175 --batch_size 64 --batch_parts 4 --model_config 'i_1_2, c_16,b,r, c_32,b,r, m_2.5_7.5, c_32,b,r, c_32,b,r, m_7.5_22.5, c_64,b,r, m_1e10_1e10, f_64,b,r,d_0.2,f_10' --fnet_widths '[16,32]' --pc_augm_scale 1.2 --pc_augm_mirror_prob 0.2 --pc_augm_input_dropout 0.1 --nworkers 3 --edgecompaction 1 --fnet_llbias 0 --odir results/modelnet10

    #CUDA_VISIBLE_DEVICES=0 python main.py --dataset modelnet40 --test_nth_epoch 25 --lr 0.1 --lr_steps '[30,60,90]' --epochs 100 --batch_size 64 --batch_parts 4 --model_config 'i_1_2, c_24,b,r, c_48,b,r, m_2.5_7.5, c_48,b,r, c_48,b,r, m_7.5_22.5, c_96,b,r, m_1e10_1e10, f_64,b,r,d_0.2,f_40' --fnet_widths '[16,32]' --pc_augm_scale 1.2 --pc_augm_mirror_prob 0.2 --pc_augm_input_dropout 0.1 --nworkers 3 --edgecompaction 1 --fnet_llbias 0 --odir results/modelnet40  
    
    #CUDA_VISIBLE_DEVICES=3 python main.py --dataset modelnet40 --test_nth_epoch 25 --lr 0.1 --lr_steps '[30,60,90]' --epochs 100 --batch_size 64 --batch_parts 8 --model_config 'i_1_2, c_24,b,r, c_48,b,r, m_2.5_7.5, c_48,b,r, c_48,b,r, m_7.5_22.5, c_96,b,r, m_1e10_1e10, f_64,b,r,d_0.2,f_40' --fnet_widths '[16,32]' --pc_augm_scale 1.2 --pc_augm_mirror_prob 0.2 --pc_augm_input_dropout 0.1 --nworkers 3 --edgecompaction 1 --fnet_llbias 0 --odir results/modelnet40_bp8
    
     
    args = parser.parse_args()
    args.start_epoch = 0
    args.lr_steps = ast.literal_eval(args.lr_steps)
    args.fnet_widths = ast.literal_eval(args.fnet_widths)
    
    logging.getLogger().setLevel(logging.INFO)
    
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
    elif args.dataset=='modelnet10' or args.dataset=='modelnet40':
        dbinfo = pointcloud_dataset.get_modelnet_info(args)
        create_dataset = pointcloud_dataset.get_modelnet
        edge_feat_func = pointcloud_dataset.cloud_edge_feats        
    else:
        raise NotImplementedError('Unknown dataset ' + rgs.dataset)
    
    if args.resume != '':
        model, optimizer = resume(args, dbinfo)
    else:
        model = create_model(args, dbinfo)   
        optimizer = create_optimizer(args, model)
    
    train_dataset = create_dataset(args, model.pyramid_conf, True)
    test_dataset = create_dataset(args, model.pyramid_conf, False)

    collate_func = functools.partial(ecc.graph_info_collate_classification, edge_func=functools.partial(edge_feat_func, args=args))
    assert args.batch_size % args.batch_parts == 0    
      
   
    def train(epoch):
        model.train()
        #TODO: set seed to dataset

        loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(args.batch_size/args.batch_parts), collate_fn=collate_func, num_workers=args.nworkers, shuffle=True)
        
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=100)

        t_loader_meter, t_trainer_meter = tnt.meter.AverageValueMeter(), tnt.meter.AverageValueMeter()
        loss_meter = tnt.meter.AverageValueMeter()
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        #cm_meter = tnt.meter.ConfusionMeter(k=dbinfo['classes'])
        
        t0 = time.time()
        
        for bidx, (inputs, targets, GIs, PIs) in enumerate(loader):
            
            t_loader = 1000*(time.time()-t0)
            


            model.set_info(GIs, PIs)
            if args.cuda: 
                inputs, targets = inputs.cuda(), targets.cuda()
        
            inputs, targets = Variable(inputs), Variable(targets)
            t0 = time.time()                
            if bidx % args.batch_parts == 0:
                optimizer.zero_grad()
                
            outputs = model(inputs)
            assert outputs.size(1)==dbinfo['classes'], "Model should output {:d} dimensions.".format(dbinfo['classes'])
            
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            
            t_trainer = 1000*(time.time()-t0)
             
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
            t_loader_meter.add(t_loader)
            t_trainer_meter.add(t_trainer)  
            logging.debug('Batch loss %f, Loader time %f ms, Trainer time %f ms.', loss.data[0], t_loader, t_trainer)
            

                
            t0 = time.time()

        return acc_meter.value()[0], loss_meter.value()[0], t_loader_meter.value()[0], t_trainer_meter.value()[0]
            
            
            
    def eval(epoch):
        model.eval()        
        #TODO: set seed to dataset
        
        if args.test_sample_avging != 'none':
            loader = torch.utils.data.DataLoader(test_dataset, batch_size=dbinfo['test_set_expansion'], collate_fn=collate_func, num_workers=args.nworkers)
        else:
            loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(args.batch_size/args.batch_parts), collate_fn=collate_func, num_workers=args.nworkers)
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG: loader = tqdm(loader, ncols=100)
        
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        cm_meter = tnt.meter.ConfusionMeter(k=dbinfo['classes'])        
        
        for bidx, (inputs, targets, GIs, PIs) in enumerate(loader):
            model.set_info(GIs, PIs)
            if args.cuda: 
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = model(inputs)
            
            if args.test_sample_avging != 'none':
                if args.test_sample_avging == 'vote':
                    outputs[2][4] = 1
                    _, ii = torch.max(outputs.data,1)
                    outputs.data.fill_(0).scatter_(1, ii, 1)
                acc_meter.add(outputs.data.mean(0), targets.data.narrow(0,1,1))
                cm_meter.add(outputs.data.mean(0), targets.data.narrow(0,1,1))                 
            else:
                acc_meter.add(outputs.data, targets.data)
                cm_meter.add(outputs.data, targets.data)  
        
        return acc_meter.value()[0], cm_meter.value()
        

        
            
            
    stats = []    
    
    for epoch in range(args.start_epoch, args.epochs):        
        print('Epoch {}/{} ({}):'.format(epoch, args.epochs, args.odir))
        
        acc_train, loss, t_loader, t_trainer = train(epoch)
        
        optimizer = lr_scheduler(optimizer, epoch, args, model)
        
        if epoch % args.test_nth_epoch == 0 or epoch==args.epochs-1:
            acc_test, cm = eval(epoch)       
            f1_test = compute_F1(cm)
            cacc_test = compute_class_acc(cm)
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
        
    with open(os.path.join(args.odir, 'trainlog.txt'), 'w') as outfile:
                    json.dump(stats, outfile)           


def resume(args, dbinfo):
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model = create_model(checkpoint['args'], dbinfo)
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
