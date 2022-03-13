

import numpy as np
import time
from sklearn.neighbors import KDTree


exp_name = 'base_10k'
lnums = 5*1000
snums = 5*1000


num_sample = 12137
num_pts = 2048
num_sp = 500
num_seg = 50
train_feat = np.load('outputs/'+ exp_name +'/models/train_feat.npy')
train_prob = np.load('outputs/'+ exp_name +'/models/train_prob.npy')
l_sp_mask = np.load('outputs/'+ exp_name +'/models/sp_mask.npy')
shape_lnum = np.sum(l_sp_mask, axis=-1)

train_sp_feat = np.zeros((num_sample, num_sp, train_feat.shape[2]), float)
train_sp_prob = np.zeros((num_sample, num_sp, train_prob.shape[2]), float)

SP_list = np.int64(np.load('../ActivePointCloud/Superpoints/super_points_list_mnfgeo500.npy'))
# SP_list = np.load('../ActivePointCloud/Superpoints/super_points_list_mnfgeo500_lamda0.01_0to12137.npy')

#### propagate super label
for i in range(num_sample):
    one_sp = SP_list[i]
    one_feat = train_feat[i]
    one_prob = train_prob[i]
    for isp in range(num_sp):
        train_sp_feat[i, isp] = one_feat[one_sp==isp].mean(axis=0)
        train_sp_prob[i, isp] = one_prob[one_sp==isp].mean(axis=0)


train_sp_feat = train_sp_feat.reshape(num_sample*num_sp, train_feat.shape[2])
train_sp_prob = train_sp_prob.reshape(num_sample*num_sp, train_prob.shape[2])
l_sp_mask = l_sp_mask.reshape(-1)

#### uncertainty
sp_etp = -np.sum(train_sp_prob*np.log(train_sp_prob+1e-6), axis=-1)
#### shape diversity
shape_div = np.log(1/(shape_lnum+1)+0.1)


#### diversity
# train_idx = np.arange(num_sample*num_sp)
# lidx = train_idx[l_sp_mask==1]
# uidx = train_idx[l_sp_mask==0]
# l_feat = train_sp_feat[l_sp_mask==1]
# u_feat = train_sp_feat[l_sp_mask==0]
# # time1 = time.time()
# tree = KDTree(l_feat, leaf_size=2)
# dist, ind = tree.query(u_feat, k=1)
# # time2 = time.time()
# # print('time', time2 - time1)
# ressp = []

beta = 0.2
alpha = 0.1
for i in range(snums):
    if i%1000==0:
        train_idx = np.arange(num_sample*num_sp)
        lidx = train_idx[l_sp_mask==1]
        uidx = train_idx[l_sp_mask==0]
        l_feat = train_sp_feat[l_sp_mask==1]
        u_feat = train_sp_feat[l_sp_mask==0]
        tree = KDTree(l_feat, leaf_size=2)
        dist, ind = tree.query(u_feat, k=1)
        fnscore = (1.0-beta)*dist + beta*sp_etp + alpha*shape_div
        ressp = []
    while True:
        m_idx = np.argmax(dist)
        if ind[m_idx] in ressp:
            dist[m_idx] = 0
        else:
            ressp.append(ind[m_idx])
            l_sp_mask[uidx[m_idx]] = 1
            break

print('l_sp_mask', np.sum(l_sp_mask))
np.save('outputs/'+ exp_name +'/models/sp_mask'+ np.str(np.sum(l_sp_mask)) +'.npy', l_sp_mask.reshape(num_sample, num_sp))

