# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:48:37 2018

@author: kai
"""

"""
sparse fully connected network for visual encoding
use network to simulate [sparse optimization method OMP]
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.autograd import Variable
import cv2

from optparse import OptionParser
from GallantData import GallantDataset
import numpy as np
from util import *
import os
import visdom



class SFCLinear(nn.Module):
    def __init__(self, in_features, out_features, c, h, w, c1):
        super(SFCLinear, self).__init__()
            
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))
        
        self.w.data.uniform_(-0.1, 0.1)
        self.b.data.uniform_(-0.1, 0.1)
        
    def forward(self, x):
        x = x.mm(self.w**2)
        return x + self.b


class FC_inverse(nn.Module):
    def __init__(self, size, base_channel=64, block_num=3, stride=2, num=1294, selected_channel=64, gpu=True):
        super(FC_inverse, self).__init__()

        out_channel = 1**(block_num-1)*base_channel
        
        h = size//(stride**(block_num))
        w = size//(stride**(block_num))
        out_dims = h*w*out_channel

        self.fc_inverse = nn.Linear(num, out_dims)
    
    def forward(self, v):
        v = self.fc_inverse(v)
        return v


class FC_direct(nn.Module):
    def __init__(self, num=1294):
        super(FC_direct, self).__init__()
        
        self.w = nn.Parameter(torch.randn(num))
        self.b = nn.Parameter(torch.randn(num))
        
    def forward(self, x):
        x = x*self.w + self.b
        return x
    
    
class ConvBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel, stride):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
#            nn.Conv2d(inchannel, outchannel, kernel_size=kernel, stride=1, padding=(kernel-1)//2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(outchannel, outchannel, kernel_size=kernel, stride=1, padding=(kernel-1)//2),
#            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel, stride=stride, padding=(kernel-1)//2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out = self.layer(x)
        return out
    
    
class SFCNet(nn.Module):
    def __init__(self, size, base_channel=64, block_num=3, kernel=3, stride=2, num=1294, selected_channel=64):
        super(SFCNet, self).__init__()
        self.inchannel = base_channel
        
        self.conv1 = nn.Conv2d(1, base_channel, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer = self.make_layer(ConvBlock, base_channel, block_num, kernel, stride)
        
        out_channel = 1**(block_num-1)*base_channel
        
        h = size//(stride**(block_num+1))
        w = size//(stride**(block_num+1))
        out_dims = h*w*self.inchannel
        print(h, w, self.inchannel, out_dims, num)
        self.fc = SFCLinear(out_dims, num, base_channel, h, w, selected_channel)
        
    def make_layer(self, block, channels, num_blocks, kernel, stride):
        layers = []
        for i in range(num_blocks):
            if i==(num_blocks-1):
                layers.append(block(self.inchannel, channels, kernel, stride))
            else:
                layers.append(block(self.inchannel, channels, kernel, stride))
            self.inchannel = channels
            channels = channels*1
            
        return nn.Sequential(*layers)

    def forward(self, x):   
        x = self.relu(self.conv1(x))
        x = self.layer(x)
        x = x.view(x.shape[0], -1)      
        x = self.fc(x)
        return x
    
    def features(self, x):
        x = self.relu(self.conv1(x))
        x = self.layer(x)
        return x


def train_net(net,
              epochs,
              bs,
              lr,
              dataset_trn,
              dataset_val,
              k,
              num, 
              viz,
              area,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
        selected voxel num: {}
    '''.format(epochs, bs, lr, len(dataset_trn),
               len(dataset_val), str(save_cp), str(gpu), str(k)))

    loader_trn = torch.utils.data.DataLoader(dataset_trn, 
                                             batch_size=bs,
                                             shuffle=True,
                                             num_workers=2,
                                             drop_last=True)
    
#    optimizer = optim.SGD(net.parameters(),
#                          lr=0.01,
#                          momentum=0.9,
#                          weight_decay=1e-6)

    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           weight_decay=1e-6)


    best_cor = 0.0
    trace = dict(title=title,
                 xlabel = 'epoch',
                 ylabel = 'correlation',
                 legend = ['train', 'train_topk', 'test', 'test_topk'],
                 markersymbol = 'dot')
    
    trace_mse = dict(title=title_mse,
                     xlabel = 'epoch',
                     ylabel = 'mse',
                     legend = ['train', 'train_topk', 'test', 'test_topk'],
                     markersymbol = 'dot')
    
    trace_bar = dict(stacked=True,
                     legend=['train', 'test'])
    
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        net.train()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        
        for i, data in enumerate(loader_trn):
            imgs, labels, voxels = data
            imgs = Variable(imgs).float()
            labels = Variable(labels)
            voxels = Variable(voxels).float()
            
            if gpu:
                imgs = imgs.cuda()
                labels = labels.cuda()
                voxels = voxels.cuda()
            
            pred_v = net(imgs)
            
            pred_v = pred_v + 1.0*Variable(torch.randn(pred_v.shape)).cuda()

            loss = loss_w(pred_v, voxels) + 1e-4*torch.mean(pred_v)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        corVector_trn, cor_trn, cor_topk_trn, index_cor, mseVector, mse_trn, mse_topk_trn, index_mse = eval_model(net, dataset_trn, 70, k)
        corVector_val, cor_val, corptopk_val, index_cor, mseVector, mse_val, mseptopk_val, index_mse = eval_model(net, dataset_val, 60, k, index_cor, index_mse)
        
        print('\tloss:%.3f'%(loss.cpu().data[0]))
        print('\ttrain cor:%.3f, val cor:%.3f'%(cor_trn, cor_val))
        print('\ttrain topk:%.3f, val ptopk:%.3f'%(cor_topk_trn, corptopk_val))
        mse_trn = np.log10(mse_trn)
        mse_val = np.log10(mse_val)
        mse_topk_trn = np.log10(mse_topk_trn)
        mseptopk_val = np.log10(mseptopk_val)
        colormaps = ['Viridis',"RdBu",'Greys','YIGnBu','Greens','YIOrRd','Bluered','Picnic','Portland','Jet','Hot','Blackbody','Earth','Electric']
        
        if epoch==0:
            win = viz.line(X=np.array([epoch]), Y=np.column_stack((np.array(cor_trn),np.array(cor_topk_trn),\
                                      np.array(cor_val),np.array(corptopk_val))), opts=trace)
            win_mse = viz.line(X=np.array([epoch]), Y=np.column_stack((np.array(mse_trn),np.array(mse_topk_trn),
                                          np.array(mse_val),np.array(mseptopk_val))), opts=trace_mse) 
            
            bar_trn = viz.bar(X=corVector_trn)
            bar_val = viz.bar(X=corVector_val)
            
        else:
            viz.line(X=np.array([epoch]), Y=np.column_stack((np.array(cor_trn),np.array(cor_topk_trn),\
                                np.array(cor_val),np.array(corptopk_val))), win=win, opts=trace, update='append')
                
            viz.line(X=np.array([epoch]), Y=np.column_stack((np.array(mse_trn),np.array(mse_topk_trn),\
                                np.array(mse_val),np.array(mseptopk_val))), win=win_mse, opts=trace_mse, update='append')
            
            viz.bar(X=corVector_trn, win=bar_trn)
            viz.bar(X=corVector_val, win=bar_val)
        
        if corptopk_val>best_cor:
            best_cor = corptopk_val
            torch.save(net.state_dict(), 'model/' + area + '_C_best.pth')
            print('Checkpoint {} saved!'.format(epoch + 1))
            file = 'results/COR/' + area + '_corVector_trn.npy'
            np.save(file, corVector_trn)
            file = 'results/COR/' + area + '_corVector_val.npy'
            np.save(file, corVector_val)     
            
#        if epoch % 10 == 0:
#            for param_group in optimizer.param_groups:
#                param_group['lr'] *= 1/10
 
    
def compute_mse(net, net1, dataset):
    net1.eval()
    loader = torch.utils.data.DataLoader(dataset,batch_size=60,shuffle=False,num_workers=2,drop_last=False)
    l = 0.0
    count = 0
    criterion = nn.MSELoss()
    for i, data in enumerate(loader):
        imgs, labels, voxels = data   
        imgs = Variable(imgs).float()
        labels = Variable(labels)
        voxels = Variable(voxels).float()
        
        imgs = imgs.cuda()
        labels = labels.cuda()
        voxels = voxels.cuda()
        
        v = net1(net(imgs))
        loss = criterion(v, voxels)
#        l = l + loss.cpu().data[0]
        l = l + loss.cpu().item()
        count = count + 1
        
    l = l/count   
    net1.train()
    return l
  

def cor_v(x1, x2):
    x1_mean, x1_var = torch.mean(x1, 0), torch.var(x1, 0)
    x2_mean, x2_var = torch.mean(x2, 0), torch.var(x2, 0)
    corVector = torch.mean((x1-x1_mean)*(x2-x2_mean), 0)
    corVector = corVector/(1e-6+torch.sqrt(x1_var*x2_var))
    return corVector


def mse_v(x1, x2):
    mseVector = torch.mean((x1-x2)**2, 0)
    return mseVector


def cor(x1, x2):
    return torch.mean(cor_v(x1, x2))


def loss_w(x1, x2):
    corVector = cor_v(x1, x2)
#    loss1 = torch.mean(corVector)
    loss2 = torch.mean(corVector**3)
    return -loss2


def topk(x, k):
    corTopkv, corTopki = torch.topk(x, k)
    corTopk = torch.mean(corTopkv)
    return corTopk, corTopki


def ptopk(x, ix):
    numsv = ix.shape[0]
    tmp = 0
    for i in range(numsv):
        tmp = tmp + x[ix[i]]
    tmp = tmp/int(numsv)
    return tmp


def eval_model(net, dataset, bs, k, index_cor=None, index_mse=None):
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=bs,
                                         shuffle=dataset,
                                         num_workers=0,
                                         drop_last=False)
    ##-------------------------------------------------------------------------
    cor = 0.0
    mse = 0.0
    corVector = 0
    mseVector = 0
    count = 0
    net.eval()
    for i, data in enumerate(loader):
        imgs, labels, voxels = data
        imgs = Variable(imgs).float().cuda()
        labels = Variable(labels).cuda()
        voxels = Variable(voxels).float().cuda()
        
        pred = net(imgs)
        
        cor_v_ = cor_v(pred, voxels)    
        mse_v_ = mse_v(pred, voxels)
        cor = cor + cor_v_.mean().data[0] 
        mse = mse + mse_v_.mean().data[0] 
        
        corVector = corVector + cor_v_.data
        mseVector = mseVector + mse_v_.data
        count = count + 1
    
    cor = cor/count
    mse = mse/count
    corVector = corVector/count
    mseVector = mseVector/count
    
    if index_cor is not None:
        cor_topk = ptopk(corVector, index_cor)
        mse_topk = ptopk(mseVector, index_cor)
    else:
        cor_topk, index_cor = topk(corVector, k)
        mse_topk, index_mse = topk(-mseVector, k)
        mse_topk = -1*mse_topk
        
    corVector = corVector.cpu().numpy()
    mseVector = mseVector.cpu().numpy()
    ##-------------------------------------------------------------------------
    return corVector, cor, cor_topk, index_cor, mseVector, mse, mse_topk, index_mse


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=50, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=64,
                      type='int', help='batch size')
    parser.add_option('-n', '--num', dest='num', default=300,
                      type='int', help='voxel number')
    parser.add_option('-l', '--learning-rate', dest='lr', default=1e-3,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('--data_dir', type=str, default='/home/amax/QK/Gallant_data')
    parser.add_option('--gpu_num', type=str, default='1')    
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    
    root_trn = '/home/amax/QK/Gallant_images/image_500/stimtrn/'
    datatxt_trn = '/home/amax/QK/Gallant_images/label/qk/stimtrn.txt'
    root_val = '/home/amax/QK/Gallant_images/image_500/stimval/'
    datatxt_val = '/home/amax/QK/Gallant_images/label/qk/stimval.txt'

    area = 'v1'
    vs_trn = np.load(os.path.join(args.data_dir, '%strn_s1.npy'%(area)))
    vs_val = np.load(os.path.join(args.data_dir, '%sval_s1.npy'%(area)))
    vs_trn = np.asarray(vs_trn, dtype=np.float64)
    vs_val = np.asarray(vs_val, dtype=np.float64)
    
    num_v = vs_trn.shape[1]
    print('the number voxels of', area, ':', num_v)
    
    size = 128
    gallantdataset_trn = GallantDataset(root_trn, datatxt_trn, vs_trn, size)
    gallantdataset_val = GallantDataset(root_val, datatxt_val, vs_val, size)

    base_channel = 128
    block_num = 2
    kernel = 3
    selected_channel = 128
    stride = 2
    net = SFCNet(size, base_channel, block_num, kernel, stride, num_v, selected_channel)
    print(net.parameters)

    title = 'cor/ROI=%s/c=%d/sc=%d/bn=%d'%(area, base_channel, selected_channel, block_num)  
    title_mse = 'mse/ROI=%s/c=%d/sc=%d/bn=%d'%(area, base_channel, selected_channel, block_num) 
    viz_name = 'E_' + area
    viz = visdom.Visdom(env=viz_name)
    
    trace = dict(title=title,
                 xlabel = 'epoch',
                 ylabel = 'correlation',
                 legend = ['train', 'train_topk', 'test', 'test_topk'],
                 markersymbol = 'dot')
    
    trace_mse = dict(title=title_mse,
                 xlabel = 'epoch',
                 ylabel = 'mse',
                 legend = ['train', 'train_topk', 'test', 'test_topk'],
                 markersymbol = 'dot')
    
    if args.load:
        net = SFCNet(size, base_channel, block_num, kernel, stride, num_v, selected_channel)
        model_path = 'model/' + area + '_C_best.pth'
        net.load_state_dict(torch.load(model_path))
        print('Model loaded from {}'.format(model_path))
        
        net1 = FC_direct(num_v)
        visualize(size, net, net1, viz, gallantdataset_trn, gallantdataset_val, num_v, area, True)
        
    else:
        if args.gpu:
            net.cuda()
            # cudnn.benchmark = True # faster convolutions, but more memory
    
        try:
            train_net(net,
                      args.epochs,
                      args.batchsize,
                      args.lr,
                      gallantdataset_trn,
                      gallantdataset_val,
                      args.num,
                      num_v,
                      viz,
                      area,
                      gpu=args.gpu,
                      img_scale=args.scale)
        
        
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
            
            


        

        

