'''
Rember to change args.teach_path for different teacher settings. 
In our experimental settings, we have the following teachers:
best_acc.*   best_loss.*   
epoch_199.*  epoch_178.*  epoch_158.*  epoch_118.* epoch_78.* epoch_50.* epoch_4.* 
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as T
import pandas as pd
import numpy as np
import os
import copy
import argparse
import random
from utils import *


def parse():
    parser = argparse.ArgumentParser(description='Generate learning path for CIFAR10/100')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--scheduler',default='cosine',type=str,help='cosine or multi')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--net',default='resnet18',type=str,
                                help='resnet18, resnet50, efficientb3 (with pre-train), mobile, vgg')
    parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10, cifar100')
    parser.add_argument('--ny_ratio', default=0.,type=float)
    parser.add_argument('--batch_size',default=256, type=int)
    parser.add_argument('--seed',default=10086,type=int)
    parser.add_argument('--proj_name',default=None, type=str)
    parser.add_argument('--run_name',default=None, type=str)
    parser.add_argument('--num_work',default=4, type=int)
    parser.add_argument('--smoothing',default=0.05, type=float,help='smoothing factor in FilterBAN')
    
    parser.add_argument('--tau',default=1., type=float,help='temperature in all KD')
    parser.add_argument('--teach_path', default='resnet18cifar100_cosine',type=str)
    parser.add_argument('--teach_type',default='net', type=str, help='net or table')
    parser.add_argument('--teach_name',default=None,help='the path and name of the teacher')
    args = parser.parse_args()
    if args.proj_name is None:
        args.proj_name = 'Distill_'+args.teach_path
    if args.dataset=='cifar10':
        args.k_clas = 10
    elif args.dataset=='cifar100':
        args.k_clas = 100
    return args

def get_validation(model, data_loader):
    model.eval()
    b_cnt, correct = 0, 0
    valid_loss, pb_table, tf_table = [],[],[]
    batch_size = data_loader.batch_size
    for x, _, ny, idx in data_loader:
        b_cnt += 1
        x,ny = x.float().cuda(), ny.long().cuda()
        with torch.no_grad():
            hid = model(x)

            loss = nn.CrossEntropyLoss()(hid, ny.squeeze())
            valid_loss.append(loss.item())
            hid = hid.detach()
            pred_idx = hid.data.max(1, keepdim=True)[1]
            prob = torch.gather(nn.Softmax(1)(hid),dim=1, index=pred_idx)
            pb_table.append(prob)
            tf_table.append(pred_idx.eq(ny.data.view_as(pred_idx)))
    model.train()
    pb_table = torch.stack(pb_table).reshape(-1,1)
    tf_table = torch.stack(tf_table).reshape(-1,1)
    ECE = cal_ECE(pb_table, tf_table)
    B_NUM = batch_size*b_cnt
    correct = tf_table.sum()
    return correct/B_NUM, np.mean(valid_loss), ECE

def train(model, optimizer, scheduler, loss_type='from_oht', teacher=None, teach_table=None, tau=1):
    results = {'tacc':[], 'vacc':[], 'tloss':[],'vloss':[],'tECE':[],'vECE':[], 'bestg_ac':[],'bestg_lo':[]}
    vacc_max, vloss_min = 0, 50
    bestg_ac, bestg_lo = 0, 0
    ES_Model = copy.deepcopy(model)
    if teach_table is not None:
        teach_table = teach_table.cuda()
        
    for g in range(args.epochs):
        for x, _, ny, idx in train_loader:
            model.train()
            x,ny = x.float().cuda(), ny.long().cuda()
            optimizer.zero_grad()
            hid = model(x)

            if teacher!=None:
                teacher.eval()
                hid_teach = teacher(x)
                hid_teach = hid_teach.detach()
            if loss_type=='from_oht':
                loss = nn.CrossEntropyLoss()(hid, ny.squeeze())
            elif loss_type=='from_teach_net':
                loss = distil_loss(hid, ny, hid_teach, T=tau, alpha=1)
            elif loss_type=='from_teach_table':
                teacher_score = teach_table[idx]
                teacher_score = add_tau_to_p(teacher_score,tau)
                loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(hid/tau, 1), teacher_score)*(tau*tau * 2.0)
            loss.backward()
            optimizer.step()
            wandb.log({'loss':loss.item()})
        # ----- At the end of each epoch --------
        scheduler.step()
        wandb.log({'learning_rate':optimizer.param_groups[0]['lr']})
        tacc, tloss, tECE = get_validation(model, data_loader=train_loader)
        vacc, vloss, vECE = get_validation(model, data_loader=valid_loader)
        if vloss<vloss_min:
            vloss_min = vloss
            bestg_lo = g
        if vacc>vacc_max:
            vacc_max = vacc         
            bestg_ac = g        
        results['tacc'].append(tacc.item())
        results['vacc'].append(vacc.item())
        results['tloss'].append(tloss)
        results['vloss'].append(vloss)
        results['tECE'].append(tECE)
        results['vECE'].append(vECE)
        results['bestg_ac'].append(bestg_ac)
        results['bestg_lo'].append(bestg_lo)
        wandb_record_results(results, g)

def main():
    global args, device, train_loader, valid_loader, BATCH_X
    global PATH_TRAIN_COARSE, PATH_BATCH_COARSE, PATH_BATCH_FINE
    global save_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse()
    rnd_seed(args.seed)
    # -------- Initialize wandb
    run_name = wandb_init(proj_name=args.proj_name, run_name=args.run_name, config_args=args)
    #run_name = 'add'
    save_path = './results/'+args.proj_name+'/'+args.teach_type+'_'+args.teach_name+'_'+run_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # -------- Prepare loader, model, optimizer, etc.
    train_loader, valid_loader = data_gen(args, valid_split=False)    
    net = get_init_net(args, args.net)
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
    if args.scheduler=='cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    elif args.scheduler=='multi':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160, 180], gamma=0.1)

    # -------- Train the model and record the path
    if args.teach_type=='net':     # This is normal KD
        teach_net = get_init_net(args, args.net)
        teach_net.to(device)
        tmp_path = os.path.join('./results/Gen_teacher', args.teach_path,args.teach_name+'.pt')
        teach_net.load_state_dict(torch.load(tmp_path))
        train(net,optimizer,scheduler,'from_teach_net',teacher=teach_net,tau=args.tau)
    elif args.teach_type=='table': # This is FilterKD
        tmp_path = os.path.join('./results/Gen_teacher', args.teach_path,args.teach_name+'.npy')
        teach_table = torch.from_numpy(np.load(tmp_path))
        train(net,optimizer,scheduler,'from_teach_table',teach_table=teach_table,tau=args.tau)
    else:
        print('args.teach must be net or table')
        
if __name__ == '__main__':
    main()
