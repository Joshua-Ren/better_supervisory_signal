'''
    This is the old verison (CIFAR100) of experiments on noisy-label case.
    Different from experiments using main_gen_teacher.py + main_distill.py,
    in this file we use the following 2 designs to ensure the only difference
    between OHT, ESKD, FilterKD to be p_tar:
    1. We initialize the OHT_model, then copy it two times to ensure the same
       initialization parameters
    2. Writting them in the same .py file can ensure all the methods using the
       same dataloader, in which the flipped labels are the same.
    
    In this old version, we do not use wandb to track the results, all the necessary
    evaluation output are stored in a *.txt file in specific folders.

    Here is the bash file we used for generating experiments:
    for n in 0 1 2 3 4 5 6 7 8 9
    do
        for t in 10 4 2 1 0.5
        do
            run python old_cifar_noisy.py --temp $t --noisy_level $n
        done
    done
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import argparse
import random
from utils import *
import copy

from models import *

EPOCHS = 100
K_CLAS = 100
NOISY_TABLE = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch_size',default=128, type=int)
parser.add_argument('--seed',default=10086,type=int)
parser.add_argument('--path',default='test', type=str)
parser.add_argument('--smoothing',default=0.05, type=float,help='smoothing factor in FilterBAN')
parser.add_argument('--noisy_level',default=0,type=int,help='level of noise imposed on label')
parser.add_argument('--temp', default=1.0, type=float, help='temperature for distilling')

args = parser.parse_args()

noisy_ratio = NOISY_TABLE[args.noisy_level]
time_name = time.asctime()[4:7]+time.asctime()[8:10]+'_'+time.asctime()[11:13]+time.asctime()[14:16]
save_path = './results/noisy_cifar100_notempsquare/run_'+time_name+str(args.batch_size)+'_'+str(args.lr)+'_'+str(args.smoothing)+'_n'+str(args.noisy_level)
if not os.path.exists(save_path):
    os.makedirs(save_path)


# ======== Set random seed ========================
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======== Get Loader with Index ===================
origin_train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=transforms.ToTensor()),batch_size=50000, shuffle=False, drop_last=True)

origin_test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=transforms.ToTensor()),batch_size=10000, shuffle=False, drop_last=True)

for train_x, train_y in origin_train_loader:
    break
for test_x, test_y in origin_test_loader:
    break

class MyDataset(Data.Dataset):
    def __init__(self, x, y, ny, transform=None,):
        self.x = x
        self.y = y
        self.ny = ny
        self.transform=transform

    def __getitem__(self,index):
        img, target, n_target, idx = self.x[index], self.y[index], self.ny[index], index
        
        if self.transform is not None:
            img = self.transform(img)       
        return img, target, n_target, idx

    def __len__(self):
        return self.y.shape[0]

def data_gen(train_x, train_y, test_x, test_y, noisy_ratio=0.1):
    train_transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])
    test_transform =transforms.Compose([
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])
    # Pack the train_loader
    train_ny = add_noise_to_y(train_y,num_clas=K_CLAS,noisy_ratio=noisy_ratio)
    dataset_train = MyDataset(train_x[:45000], train_y[:45000], train_ny[:45000], train_transform)
    dataset_valid = MyDataset(train_x[45000:], train_y[45000:], train_ny[45000:], train_transform)
    train_loader = Data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=2)
    valid_loader = Data.DataLoader(dataset_valid, batch_size=5000, shuffle=True, drop_last = True, num_workers=2)
    # Pack the test_loader 
    dataset_test = MyDataset(test_x, test_y, test_y, test_transform)
    test_loader = Data.DataLoader(dataset_test, batch_size=1000, shuffle=False, drop_last = True, num_workers=2)
    return train_loader,valid_loader, test_loader

# ======== Get Model ===================
def get_init_net():
    net = ResNet18(num_classes=K_CLAS)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    return net

# =========== Record the paths ================
def _Update_Teach_Table(store_table,pred_batch, idx_batch):
    pred_items = pred_batch.cpu().detach()
    batch_size = idx_batch.shape[0]
    for i in range(batch_size):
        idx_key = idx_batch[i].item()
        tmp_pred = pred_items[i]
        if store_table[idx_key, :].sum()==0:
            store_table[idx_key, :] = tmp_pred
        else:
            store_table[idx_key, :] = (1-args.smoothing)*store_table[idx_key, :] + args.smoothing*tmp_pred

# ========== Train and Validatoin Function ======
def get_validation(model, data_loader, epoch=None, store_table=None):
    model.eval()
    b_cnt, correct = 0, 0
    valid_loss, pb_table, tf_table = [],[],[]
    batch_size = data_loader.batch_size
    for x, _, ny, idx in data_loader:
        b_cnt += 1
        x,ny = x.float().cuda(), ny.long().cuda()
        with torch.no_grad():
            hid = model(x)
            if store_table is not None:
                pred_batch = F.softmax(hid, 1)
                _Update_Teach_Table(store_table, pred_batch, idx)

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

def train(model, optimizer, scheduler, loss_type='from_oht',teacher=None, teach_table=None, kd_temperature=1, store_table=None,file_name='test'):
    results = {'tacc':[], 'vacc':[], 'test_acc':[], 'tloss':[],'vloss':[],'tECE':[],'vECE':[],'test_ECE':[]}       
    vacc_max = 0
    vloss_min = 50
    file = save_path+'/'+file_name+'.txt'
    with open(file, 'w') as f:
        f.write('======lr \t{:.6f}\n'.format(args.lr))
        f.write('======bsize \t{:5d}\n'.format(args.batch_size))
        f.write('======seed \t{:5d}\n'.format(args.seed))
        f.write('======smoth \t{:.6f}\n'.format(args.smoothing))
        f.write('======temp\t{:.6f}\n'.format(kd_temperature))
        f.write('======Noisy ratio is \t{:4f}\n'.format(noisy_ratio))
    best_store_table = copy.deepcopy(store_table)
    ES_Model = copy.deepcopy(model)
    if teach_table is not None:
        teach_table = teach_table.cuda()
    for g in range(EPOCHS):
        for x,_,ny,idx in train_loader:
            x,ny = x.float().cuda(), ny.long().cuda()
            optimizer.zero_grad()
            hid = model(x)

            if store_table is not None:
                pred_batch = F.softmax(hid/kd_temperature, 1)
                _Update_Teach_Table(store_table, pred_batch, idx)

            if teacher!=None:
                teacher.eval()
                hid_teach = teacher(x)
                hid_teach = hid_teach.detach()
            if loss_type=='from_oht':
                loss = nn.CrossEntropyLoss()(hid, ny.squeeze())
            elif loss_type=='normal_kd':
                loss = distil_loss(hid, ny, hid_teach, T=kd_temperature, alpha=1)
            elif loss_type=='from_teach_table':
                teacher_score = teach_table[idx]
                loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(hid/kd_temperature, 1), teacher_score)*(kd_temperature*kd_temperature * 2.0)
            loss.backward()
            optimizer.step()
       # ----- At the end of each epoch --------
        tacc, tloss, tECE = get_validation(model, data_loader=train_loader)
        vacc, vloss, vECE = get_validation(model, data_loader=valid_loader)
        test_acc, _, test_ECE = get_validation(model, data_loader=test_loader)
        results['tacc'].append(tacc.item())
        results['vacc'].append(vacc.item())
        results['test_acc'].append(test_acc.item())
        results['tloss'].append(tloss)
        results['vloss'].append(vloss)
        results['tECE'].append(tECE)
        results['vECE'].append(vECE)
        results['test_ECE'].append(test_ECE)
        #if vloss<vloss_min:
            #vloss_min = vloss
        if vacc>vacc_max:
            vacc_max = vacc         
            idx_max = g
            best_store_table = copy.deepcopy(store_table)
            ES_Model = copy.deepcopy(model)
        scheduler.step()
        if g%5==1:
            with open(file, 'a') as f:
                f.write('Epoch: {:3d}\tTLOSS: {:.6f}\tTACC: {:.6f},\tTECE:{:.6f}, \tVLOSS: {:.6f},\tVACC:{:.6f}, \tVECE:{:.6f},\tGACC:{:.6f},\tIDX:{:3d}\n'.format(g,tloss,tacc,tECE,vloss,vacc,vECE,test_acc,idx_max))
            print('\t==Training , Epoch: {:3d}/{:3d}\tLoss: {:.6f}\tTACC: {:.6f},\tVACC:{:.6f}, ES_ID: {:3d}'.format(g,EPOCHS, tloss, tacc, vacc,idx_max))
    return ES_Model, results, best_store_table        

# ======== Run ========
train_loader, valid_loader, test_loader = data_gen(train_x, train_y, test_x, test_y, noisy_ratio=noisy_ratio)

    # ---- First run on OHT -------
OHT_Table = torch.zeros((45000, K_CLAS))
tmp_file = 'OHT'
OHT_model = get_init_net()
stud1_model = copy.deepcopy(OHT_model)
stud2_model = copy.deepcopy(OHT_model)

OHT_optimizer = optim.SGD(OHT_model.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
OHT_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(OHT_optimizer, T_max=EPOCHS, eta_min=1e-5)
ES_model, OHT_results, ES_OHT_table = train(OHT_model, OHT_optimizer, OHT_scheduler, 'from_oht', 
                                            kd_temperature=args.temp, store_table=OHT_Table, file_name=tmp_file)


    # ------- KD from ES_model ------
tmp_file = 'ESKD'
teach_model = ES_model
stud1_Table = torch.zeros((45000, K_CLAS))
stud1_optimizer = optim.SGD(stud1_model.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
stud1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stud1_optimizer, T_max=EPOCHS, eta_min=1e-5)
_, stud1_results, ES_stud1_Table = train(stud1_model, stud1_optimizer, stud1_scheduler, 'normal_kd', 
                                        kd_temperature=args.temp,teacher=teach_model, store_table=stud1_Table, file_name=tmp_file)



    # -------- Filter KD ------------
tmp_file = 'Filter'
teach_Table = ES_OHT_table
stud2_Table = torch.zeros((45000, K_CLAS))
stud2_optimizer = optim.SGD(stud2_model.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
stud2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stud2_optimizer, T_max=EPOCHS, eta_min=1e-5)
_, stud2_results, ES_stud2_Table = train(stud2_model, stud2_optimizer, stud2_scheduler, 'from_teach_table', 
                                        kd_temperature=args.temp,teach_table=teach_Table, store_table=stud2_Table, file_name=tmp_file)


