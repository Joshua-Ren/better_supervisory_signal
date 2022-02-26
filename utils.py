'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import random
import math
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
from models import *

def save_checkpoint(model, save_path, file_name='test'):
    file_path = os.path.join(save_path, file_name+'.pt')
    torch.save(model.state_dict(), file_path)
    
def save_storetable(table, save_path, file_name='test'):
    file_path = os.path.join(save_path, file_name+'.npy')
    np.save(file_path, table.numpy())

def rnd_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# =========== wandb functions =================
def wandb_init(proj_name='test', run_name=None, config_args=None):
    wandb.init(
        project=proj_name,
        config={})
    if config_args is not None:
        wandb.config.update(config_args)
    if run_name is not None:
        wandb.run.name=run_name
        return run_name
    else:
        return wandb.run.name

def wandb_record_results(results, epoch):
  for key in results.keys():
    wandb.log({key:results[key][-1]})
  wandb.log({'epoch':epoch})

# ======== Get Model ===================
def get_init_net(args, net_type):
    if net_type=='resnet18':
        net = ResNet18(args.k_clas)
    elif net_type=='resnet50':
        net = ResNet50(args.k_clas)
    elif net_type=='mobile':
        net = MobileNetV2(args.k_cla)
    elif net_type=='vgg':
        net = vgg11_bn(args.k_clas)
    elif net_type=='efficientb3':
        from efficientnet_pytorch import EfficientNet
        net = EfficientNet.from_pretrained('efficientnet-b3', num_classes=args.k_cla)
    else:
        print('net structure not supported, only support resnet18, resnet50, mobile, vgg, efficientb3')
    return net

# ======== Get Loader with Index ===================
def data_gen(args,valid_split=False):
    # ------ Write MyDataset to generate constant index and noisy-label
    class MyDataset(Data.Dataset):
        def __init__(self, x, y, ny, transform=None,):
            self.x = x
            self.y = y
            self.ny = y
            self.transform=transform
    
        def __getitem__(self,index):
            img, target, n_target, idx = self.x[index], self.y[index], self.ny[index], index
            
            if self.transform is not None:
                img = self.transform(img)       
            return img, target, n_target, idx
    
        def __len__(self):
            return self.y.shape[0]
    if args.dataset.lower()=='cifar10':
        tmp_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=T.ToTensor())
        tmp_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=T.ToTensor())        
    elif args.dataset.lower()=='cifar100':
        tmp_train = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=T.ToTensor())
        tmp_test = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=T.ToTensor())
    else:
        print('Only for cifar10 and cifar100')
    origin_train_loader = torch.utils.data.DataLoader(
                    tmp_train,batch_size=50000, shuffle=False, drop_last=True)
    origin_test_loader = torch.utils.data.DataLoader(
                    tmp_test,batch_size=10000, shuffle=False, drop_last=True)

    for train_x, train_y in origin_train_loader:
        break
    for test_x, test_y in origin_test_loader:
        break

    train_transform=T.Compose([
                       T.RandomCrop(32, padding=4),
                       T.RandomHorizontalFlip(),
                       T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])
    test_transform =T.Compose([
                       #T.RandomCrop(32, padding=4),
                       #T.RandomHorizontalFlip(),
                       T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])
    # Pack the train_loader
    train_ny = add_noise_to_y(train_y,num_clas=args.k_clas, noisy_ratio=args.ny_ratio)
    if valid_split:
        dataset_train = MyDataset(train_x[:48000], train_y[:48000], train_ny[:48000], train_transform)
        dataset_valid = MyDataset(train_x[48000:], train_y[48000:], train_ny[48000:], train_transform)
        train_loader = Data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_work)
        valid_loader = Data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_work)
        # Pack the test_loader 
        dataset_test = MyDataset(test_x, test_y, test_y, test_transform)
        test_loader = Data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_work)
        return train_loader,valid_loader, test_loader
    else:
        dataset_train = MyDataset(train_x, train_y, train_ny, train_transform)
        dataset_test = MyDataset(test_x, test_y, test_y, test_transform)
        train_loader = Data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_work)
        test_loader = Data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_work)
        return train_loader, test_loader

def cal_ECE(pb_table, tf_table):
  '''
    pb_table is the probability provided by network
    tf_table is the acc results of the prodiction
  '''
  BM_acc = np.zeros((10,))
  BM_conf = np.zeros((10,))
  BM_cnt = np.zeros((10,))
  Index_table = ((pb_table-1e-6).T*10).int().squeeze()

  for i in range(pb_table.shape[0]):
    idx = Index_table[i]
    BM_cnt[idx] += 1
    BM_conf[idx] += pb_table[i]
    if tf_table[i]:
      BM_acc[idx] += 1
  ECE = 0
  for j in range(10):
    if BM_cnt[j] != 0:
      ECE += BM_cnt[j]*np.abs(BM_acc[j]/BM_cnt[j]-BM_conf[j]/BM_cnt[j])
  return ECE/BM_cnt.sum()

def ce_loss(y, labels):
  # Cross entropy loss between y and labels
    return  nn.CrossEntropyLoss()(y, labels.squeeze())

def kd_loss(y, teacher_scores, T=1):
    return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y/T,1), F.softmax(teacher_scores/T,1))*(T*T * 2.0)

def distil_loss(y, labels, teacher_scores, T=1, alpha=0.5):
    return kd_loss(y, teacher_scores, T) * alpha + ce_loss(y, labels) * (1. - alpha)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def add_noise_to_y(y, num_clas, noisy_ratio=0.1):
    import copy
    if noisy_ratio==0:
        return y
    def _change_label(y, num_clas):
        # ------- Randomly select a label that is not same with given one ---
        # ---- y should be a torch tensor, size(1,)
        ny = torch.randint(0,num_clas,(1,))
        #while ny==y:
        #ny = torch.randint(0,num_clas,(1,))
        return ny
    N_Data = y.shape[0]
    ny = copy.deepcopy(y)
    noisy_perm = np.arange(0,N_Data,1)
    noisy_num = int(N_Data*noisy_ratio)
    np.random.shuffle(noisy_perm)
    noisy_idx = noisy_perm[:noisy_num]

    for i in range(len(noisy_idx)):
        tmp = _change_label(y[noisy_idx[i]],num_clas=num_clas)
        ny[noisy_idx[i]] = tmp
    return ny

def add_tau_to_p(p,tau):
  z = p.log()
  p_tau = F.softmax(z/tau,1)
  return p_tau