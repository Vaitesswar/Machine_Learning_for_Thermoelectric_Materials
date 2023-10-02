import argparse
import shutil
import sys
import time
import random
from random import sample
import csv
import functools
import json
import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from pymatgen.core.structure import Structure

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, optimizer, epoch, normalizer_target, normalizer_crystal):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    
    # switch to train mode
    model.train()
    
    loss_list = list()
    
    end = time.time()
    
    # Check if cuda is available
    use_cuda = True
    is_cuda = use_cuda and torch.cuda.is_available()
    torch.cuda.is_available()
    
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # normalize crystal features 
        crystal_fea = normalizer_crystal.norm(input[4])
        
        if is_cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                         Variable(crystal_fea).cuda(non_blocking=True))
        else:
            input_var = (Variable(input[0]),Variable(input[1]),input[2],input[3], Variable(crystal_fea))
                           
        # normalize target
        target_normed = normalizer_target.norm(target)
        
        target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        
        # Computing loss
        loss = cust_loss(output, target_var)
        loss_mean = loss.data.cpu()
        loss_list.append(loss_mean)
        
        #print(output,target_var,loss.data.cpu())

        # measure accuracy and record loss
        mae_error = mae(normalizer_target.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0: # print frequency = 100
            
            print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )
                
def validate(val_loader, model, normalizer_target, normalizer_crystal, test = False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
        
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            crystal_fea = normalizer_crystal.norm(input[4])
            
            if is_cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                             Variable(crystal_fea).cuda(non_blocking=True))
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3],
                             crystal_fea)

        target_normed = normalizer_target.norm(target)
        
        with torch.no_grad():
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        
        # Computing loss
        loss = cust_loss(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer_target.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        
        if test:
            test_pred = normalizer_target.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
        if i % 100 == 0:
                print('Test: [{0}/{1}]\t' # print frequency = 100
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      mae_errors=mae_errors))
            
    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    
    if test:
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,mae_errors=mae_errors))                                             
        return test_preds,test_targets
    else:
        return mae_errors.avg
    
def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target
    Parameters
    ----------
    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    prediction = 10**prediction
    target = 10**target
    errors = ((target - prediction)/target)*100
    MAE = torch.mean(torch.abs(errors))
    
    return MAE

def cust_loss(output, target_var):
    
    loss = 10**(torch.mean((target_var - output)**2))
    
    return loss