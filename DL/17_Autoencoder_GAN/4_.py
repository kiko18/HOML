# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:09:26 2021

@author: BT
"""

import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import torch
from torchvision import datasets, transforms

train_path = 'C:/Users/BT/Downloads/Train'
test_path =  'C:/Users/BT/Downloads/Test'

#mat = scipy.io.loadmat('C:/Users/BT/Downloads/train_.mat')

#X = mat['X']
#y = mat['y']

#img = X[:,:,:,0]
#plt.imshow(dat['train'][1][0].squeeze())

#----------------------------
data={}
taskcla=[]
size=[3,32,32]
nb_task = 5
nb_class_pro_task = 2

# CIFAR100
dat={}
tasknum = 50
    
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
    ])

dat['train']=datasets.Omniglot(train_path, background=True, download=True, transform=transform)
dat['test']=datasets.Omniglot(test_path, background=False, download=True, transform=transform)

ncla_dict = {}

for i in range(tasknum):
 
    ii = i#-2        
    data[ii] = {}
    data[ii]['name'] = 'omniglot-{:d}'.format(i)
    data[ii]['ncla'] = (torch.max(f['Y']['train'][i]) + 1).int().item()
    ncla_dict[ii] = data[i]['ncla']
        
    data[ii]['train'] = {'x': [], 'y': []}
    data[ii]['test'] = {'x': [], 'y': []}
    data[ii]['valid'] = {'x': [], 'y': []}

    image = f['X']['train'][i]
    target = f['Y']['train'][i]

    index_arr = np.arange(len(image))
    np.random.shuffle(index_arr)
    train_ratio = (len(image)//10)*8
    valid_ratio = (len(image)//10)*1
    test_ratio = (len(image)//10)*1
    
    train_idx = index_arr[:train_ratio]
    valid_idx = index_arr[train_ratio:train_ratio+valid_ratio]
    test_idx = index_arr[train_ratio+valid_ratio:]

    data[ii]['train']['x'] = image[train_idx]
    data[ii]['train']['y'] = target[train_idx]
    data[ii]['valid']['x'] = image[valid_idx]
    data[ii]['valid']['y'] = target[valid_idx]
    data[ii]['test']['x'] = image[test_idx]
    data[ii]['test']['y'] = target[test_idx]
    

    # "Unify" and save
    for s in ['train', 'test', 'valid']:
#                 data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
        data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
        torch.save(data[i][s]['x'],os.path.join(os.path.expanduser('../dat/binary_omniglot'), 'data' + str(i) + s + 'x.bin'))
        torch.save(data[i][s]['y'],os.path.join(os.path.expanduser('../dat/binary_omniglot'), 'data' + str(i) + s + 'y.bin'))
torch.save(ncla_dict, os.path.join(os.path.expanduser('../dat/binary_omniglot'), 'ncla_dict.pt'))
 
        