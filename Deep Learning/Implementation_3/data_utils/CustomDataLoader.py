#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import warnings
import numpy as np
import glob
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


# In[2]:


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def drop_txt(data):
    return map(lambda x: x.split('.')[0], data)


# In[10]:


class PartNormalDataset(Dataset):
    def __init__(self, root = '/home/trojan/skia_projects/3d_facial_segmentation/part_segmentation/pointnet/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                 npoints=2500, split='train', class_choice=None, normal_channel=False):

        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
    

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}

        for item, key in self.cat.items():
            train_ids = [ids for ids in os.listdir(os.path.join(self.root, key))]
            test_ids = [ids for ids in os.listdir(os.path.join(root, 'test', key))]
        vf = np.vectorize(drop_txt)
        train_ids = list(drop_txt(train_ids))
        test_ids = list(drop_txt(test_ids))
        
        for item in self.cat:        
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            #print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            #print(os.path.basename(fns))

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))
                
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
                
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
            
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Head': [0, 1]}
        
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
        
    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


# In[16]:


if __name__ == '__main__':
    d = PartNormalDataset(root = '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal', class_choice = ['Head'], split='train')
    print(len(d))
    import time
    tic = time.time()
    for i in range(len(d)):
        ps = d[i]
        print(ps)
        break



