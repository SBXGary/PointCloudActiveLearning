#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@Time: 2021/7/20 7:49 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
# import cv2
from torch.utils.data import Dataset
import random
from scipy import stats


def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('modelnet40_ply_hdf5_2048', DATA_DIR))
        os.system('rm %s' % (zipfile))


def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('indoor3d_sem_seg_hdf5_data', DATA_DIR))
        os.system('rm %s' % (zipfile))
    if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')):
        if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')):
            print('Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
                from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
            sys.exit(0)
        else:
            zippath = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')
            os.system('unzip %s' % (zippath))
            os.system('mv %s %s' % ('Stanford3dDataset_v1.2_Aligned_Version', DATA_DIR))
            os.system('rm %s' % (zippath))


def load_data_cls(partition):
    # download_modelnet40()
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_DIR = '/shareData3/lab-shi.xian/dataset'
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    num_data = data.shape[0]
    data_idx = np.arange(0,num_data)
    return (data, label, seg, num_data, data_idx)

def load_shapenet_train():

    TRAINING_FILE_LIST = '../../dataset/ShapeNet/hdf5_data/train_hdf5_file_list.txt'
    # VAL_FILE_LIST = '../../dataset/ShapeNet/hdf5_data/val_hdf5_file_list.txt'
    # TESTING_FILE_LIST = '../../dataset/ShapeNet/hdf5_data/test_hdf5_file_list.txt'
    h5_base_path = '../../dataset/ShapeNet/hdf5_data'

    ## train/val file list
    train_file_list = getDataFiles(TRAINING_FILE_LIST)
    num_train_file = len(train_file_list)
    # val_file_list = getDataFiles(VAL_FILE_LIST)
    # num_val_file = len(val_file_list)
    # test_file_list = getDataFiles(TESTING_FILE_LIST)
    # num_test_file = len(test_file_list)


    ## train/val file index
    train_file_idx = np.arange(0,num_train_file)
    # val_file_idx = np.arange(0,num_val_file)
    # test_file_idx = np.arange(0,num_test_file)

    ## Load Train Data
    train_data = []
    train_labels = []
    train_seg = []
    num_train = 0
    # train_data_idx = []
    for cur_train_filename in train_file_list:
        print('cur_train_filename',cur_train_filename)
        cur_train_data, cur_train_labels, cur_train_seg, cur_num_train, cur_train_data_idx = loadDataFile_with_seg(
            os.path.join(h5_base_path,cur_train_filename))



        train_data.append(cur_train_data)
        train_labels.append(cur_train_labels)
        train_seg.append(cur_train_seg)
        # train_data_idx.append(cur_train_data_idx+num_train)
        num_train += cur_num_train

    train_data = np.concatenate(train_data)

    train_labels = np.concatenate(train_labels).astype(np.int64)
    train_seg = np.concatenate(train_seg).astype(np.int64)
    
    # train_data_idx = np.concatenate(train_data_idx)
    num_train = num_train


    print('len(train_data)------',train_data.shape, train_data.dtype)
    print('len(train_labels)------',train_labels.shape, train_labels.dtype)
    print('len(train_seg)------',train_seg.shape, train_seg.dtype)
    # print('len(train_data_idx)------',train_data_idx.shape)
    print('num_train------',num_train)
    

    return train_data, train_labels, train_seg

def load_shapenet_val():

    TRAINING_FILE_LIST = '../../dataset/ShapeNet/hdf5_data/train_hdf5_file_list.txt'
    VAL_FILE_LIST = '../../dataset/ShapeNet/hdf5_data/val_hdf5_file_list.txt'
    TESTING_FILE_LIST = '../../dataset/ShapeNet/hdf5_data/test_hdf5_file_list.txt'
    h5_base_path = '../../dataset/ShapeNet/hdf5_data'

    ## train/val file list
    train_file_list = getDataFiles(TRAINING_FILE_LIST)
    num_train_file = len(train_file_list)
    val_file_list = getDataFiles(VAL_FILE_LIST)
    num_val_file = len(val_file_list)
    test_file_list = getDataFiles(TESTING_FILE_LIST)
    num_test_file = len(test_file_list)


    ## train/val file index
    train_file_idx = np.arange(0,num_train_file)
    val_file_idx = np.arange(0,num_val_file)
    test_file_idx = np.arange(0,num_test_file)


    ## Load Val Data
    val_data = []
    val_labels = []
    val_seg = []
    num_val = 0
    # val_data_idx = []
    for cur_val_filename in val_file_list:
        cur_val_data, cur_val_labels, cur_val_seg, cur_num_val, cur_val_data_idx = loadDataFile_with_seg(
            os.path.join(h5_base_path, cur_val_filename))

        val_data.append(cur_val_data)
        val_labels.append(cur_val_labels)
        val_seg.append(cur_val_seg)
        # val_data_idx.append(cur_val_data_idx + num_val)
        num_val += cur_num_val

    val_data = np.concatenate(val_data)
    val_labels = np.concatenate(val_labels).astype(np.int64)
    val_seg = np.concatenate(val_seg).astype(np.int64)
    # val_data_idx = np.concatenate(val_data_idx)
    num_val = num_val


    print('len(val_data)------',val_data.shape)
    print('len(val_labels)------',val_labels.shape)
    print('len(val_seg)------',val_seg.shape)
    # print('len(val_data_idx)------',val_data_idx.shape)
    print('num_val------',num_val)
    

    return val_data, val_labels, val_seg

def load_shapenet_test():

    TRAINING_FILE_LIST = '../../dataset/ShapeNet/hdf5_data/train_hdf5_file_list.txt'
    VAL_FILE_LIST = '../../dataset/ShapeNet/hdf5_data/val_hdf5_file_list.txt'
    TESTING_FILE_LIST = '../../dataset/ShapeNet/hdf5_data/test_hdf5_file_list.txt'
    h5_base_path = '../../dataset/ShapeNet/hdf5_data'

    ## train/val file list
    train_file_list = getDataFiles(TRAINING_FILE_LIST)
    num_train_file = len(train_file_list)
    val_file_list = getDataFiles(VAL_FILE_LIST)
    num_val_file = len(val_file_list)
    test_file_list = getDataFiles(TESTING_FILE_LIST)
    num_test_file = len(test_file_list)


    ## train/val file index
    train_file_idx = np.arange(0,num_train_file)
    val_file_idx = np.arange(0,num_val_file)
    test_file_idx = np.arange(0,num_test_file)



    ## Load Test Data
    test_data = []
    test_labels = []
    test_seg = []
    num_test = 0
    # test_data_idx = []
    for cur_test_filename in test_file_list:
        print('cur_test_filename',cur_test_filename)
        cur_test_data, cur_test_labels, cur_test_seg, cur_num_test, cur_test_data_idx = load_h5_data_label_seg(
            os.path.join(h5_base_path,cur_test_filename))

        test_data.append(cur_test_data)
        test_labels.append(cur_test_labels)
        test_seg.append(cur_test_seg)
        # test_data_idx.append(cur_test_data_idx+num_test)
        num_test += cur_num_test


    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels).astype(np.int64)
    test_seg = np.concatenate(test_seg).astype(np.int64)
    # test_data_idx = np.concatenate(test_data_idx)
    num_test = num_test


    print('len(test_data)------',test_data.shape)
    print('len(test_labels)------',test_labels.shape)
    print('len(test_seg)------',test_seg.shape)
    # print('len(test_data_idx)------',test_data_idx.shape)
    print('num_test------',num_test)


    return test_data, test_labels, test_seg


def load_data_partseg(partition):
    download_shapenetpart()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def prepare_test_data_semseg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(os.path.join(DATA_DIR, 'stanford_indoor3d')):
        os.system('python prepare_data/collect_indoor3d_data.py')
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')):
        os.system('python prepare_data/gen_indoor3d_h5.py')


def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    download_S3DIS()
    prepare_test_data_semseg()
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


def load_color_partseg():
    colors = []
    labels = []
    f = open("prepare_data/meta/partseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    partseg_colors = np.array(colors)
    partseg_colors = partseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1350
    img = np.zeros((1350, 1890, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (1900, 1900), [255, 255, 255], thickness=-1)
    column_numbers = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    column_gaps = [320, 320, 300, 300, 285, 285]
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for row in range(0, img_size):
        column_index = 32
        for column in range(0, img_size):
            color = partseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.76, (0, 0, 0), 2)
            column_index = column_index + column_gaps[column]
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 50:
                cv2.imwrite("prepare_data/meta/partseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column + 1 >= column_numbers[row]):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break


def load_color_semseg():
    colors = []
    labels = []
    f = open("prepare_data/meta/semseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    semseg_colors = np.array(colors)
    semseg_colors = semseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1500
    img = np.zeros((500, img_size, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (img_size, 750), [255, 255, 255], thickness=-1)
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for _ in range(0, img_size):
        column_index = 32
        for _ in range(0, img_size):
            color = semseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.7, (0, 0, 0), 2)
            column_index = column_index + 200
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 13:
                cv2.imwrite("prepare_data/meta/semseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column_index >= 1280):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break  
    

def normal_data(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud, l1=2./3., l2=-0.2, h1=3./2., h2=0.2):
    xyz1 = np.random.uniform(low=l1, high=h1, size=[3])
    xyz2 = np.random.uniform(low=l2, high=h2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def rotate_pointcloud(pointcloud, sigma=0.01):
    angle = (2*np.random.rand() - 1) * np.pi * sigma
    direct_idx = random.sample([0,1,2], 1)

    R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    R_y = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    if direct_idx==0:
        pointcloud.dot(R_x).astype('float32')
    elif direct_idx==1:
        pointcloud.dot(R_y).astype('float32')
    elif direct_idx==2:
        pointcloud.dot(R_z).astype('float32')
    return pointcloud

def random_mirror_pointcloud(pointcloud):
    mirror_opt = np.random.choice([0,1])
    if mirror_opt == 0:
        pass
    elif mirror_opt == 1:
        pointcloud[:,2] = -pointcloud[:,2]
    return pointcloud

def pitran_pointcloud(pointcloud, jt1=0.01, jt2=0.02):
    pointcloud = jitter_pointcloud(pointcloud, sigma=jt1, clip=jt2)
    pointcloud = translate_pointcloud(pointcloud)
    pointcloud = rotate_pointcloud(pointcloud, sigma=0.01)
    pointcloud = random_mirror_pointcloud(pointcloud)
    return pointcloud

def cutout_pointcloud(pointcloud, L=0.5):
    L = L * np.random.rand()
    ord_x = 2.0 * np.random.rand() - 1.0
    ord_y = 2.0 * np.random.rand() - 1.0
    ord_z = 2.0 * np.random.rand() - 1.0
    ord_x2 = ord_x + L
    ord_y2 = ord_y + L
    ord_z2 = ord_z + L

    cut_sign = np.ones(pointcloud.shape[0], int)
    cut_sign = cut_sign * [pointcloud[:,0]>ord_x] * [pointcloud[:,0]<ord_x2] * [pointcloud[:,1]>ord_y] * [pointcloud[:,1]<ord_y2] * [pointcloud[:,2]>ord_z] * [pointcloud[:,2]<ord_z2]
    
    patch = np.clip(0.5 * np.random.randn(np.sum(cut_sign), 3), -1.0, 1.0)
    pointcloud[cut_sign[0,:]==1] = patch
    return pointcloud

def cut_one_axis(pc):
    xy_opt = np.random.choice([0,1,2,3,4,5])  #### xyz 012
    cut_range = np.clip(np.random.rand(), 0.2, 0.8)

    if (xy_opt == 0):
        pc[:, 0] = np.clip(pc[:, 0], -cut_range, 1.0)
    elif (xy_opt == 1):
        pc[:, 0] = np.clip(pc[:, 0], -1.0, cut_range)
    elif (xy_opt == 2):
        pc[:, 1] = np.clip(pc[:, 1], -cut_range, 1.0)
    elif (xy_opt == 3):
        pc[:, 1] = np.clip(pc[:, 1], -1.0, cut_range)
    elif (xy_opt == 4):
        pc[:, 2] = np.clip(pc[:, 2], -cut_range, 1.0)
    elif (xy_opt == 5):
        pc[:, 2] = np.clip(pc[:, 2], -1.0, cut_range)
    return pc




class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        # self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        

        if self.partition == 'train':
            self.data, self.label, self.seg = load_shapenet_train()
        elif self.partition == 'val':
            self.data, self.label, self.seg = load_shapenet_val()
        elif self.partition == 'test':
            self.data, self.label, self.seg = load_shapenet_test()
        print('self.data', self.data.shape)
        print('self.label', self.label.shape, np.unique(self.label), len(np.unique(self.label)))
        print('self.seg', self.seg.shape, np.unique(self.seg), len(np.unique(self.seg)))

        self.seg_num_all = 50
        self.seg_start_index = 0
        
      
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]

        indices = list(range(pointcloud.shape[0]))
        np.random.shuffle(indices)
        pointcloud = pointcloud[indices]
        seg = seg[indices]

        wa_pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
        wa_pointcloud = translate_pointcloud(wa_pointcloud)
        wa_pointcloud = jitter_pointcloud(wa_pointcloud, sigma=0.01, clip=0.02)
        wa_pointcloud = random_mirror_pointcloud(wa_pointcloud)
        return pointcloud, wa_pointcloud, seg, label

    def __len__(self):
        return self.data.shape[0]

class ShapeNetPart_SP(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        # self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        

        if self.partition == 'train':
            self.data, self.label, self.seg = load_shapenet_train()
        elif self.partition == 'val':
            self.data, self.label, self.seg = load_shapenet_val()
        elif self.partition == 'test':
            self.data, self.label, self.seg = load_shapenet_test()
        print('self.data', self.data.shape)
        print('self.label', self.label.shape, np.unique(self.label), len(np.unique(self.label)))
        print('self.seg', self.seg.shape, np.unique(self.seg), len(np.unique(self.seg)))

        self.seg_num_all = 50
        self.seg_start_index = 0
        

        if self.partition == 'train':
            ##### super_points
            self.SP_list = np.int64(np.load('../ActivePointCloud/Superpoints/super_points_list_mnfgeo500.npy'))
            # self.SP_list = np.load('../ActivePointCloud/Superpoints/super_points_list_mnfgeo500_lamda0.01_0to12137.npy')
            print('self.SP_list', self.SP_list.shape, len(np.unique(self.SP_list)))

            num_sp = 500
            num_int = 10*1000
            #### propagate mask
            self.sp_mask = np.zeros((self.data.shape[0], num_sp), int)
            print('self.sp_mask', self.sp_mask.shape)
            self.sp_mask = self.sp_mask.reshape(-1) 
            sp_idx = np.load('../ActivePointCloud/Dataset/ShapeNet/Preprocess/sp_wl_pick_list_sp500.npy')
            print('sp_idx', sp_idx.shape, len(np.unique(sp_idx)))
            self.sp_mask[sp_idx[:num_int]] = 1
            self.sp_mask = self.sp_mask.reshape(self.data.shape[0], num_sp) 
            print('self.sp_mask', self.sp_mask.shape, np.sum(self.sp_mask))
            self.pts_mask = np.zeros((self.data.shape[0], self.data.shape[1]), int)

            # #### propagate super label
            self.sp_seg = self.seg.copy()
            for i in range(self.SP_list.shape[0]):
                one_sp = self.SP_list[i]
                one_seg = self.sp_seg[i]
                for isp in range(num_sp):
                    one_mask_idx = np.arange(self.SP_list.shape[1])[one_sp==isp]
                    mask_seg = one_seg[one_mask_idx]
                    one_seg[one_mask_idx] = stats.mode(mask_seg)[0]

                    if self.sp_mask[i, isp]==1:
                        self.pts_mask[i, one_mask_idx] = 1

                self.sp_seg[i] = one_seg
                # print('eeerror', np.sum((self.sp_seg[i]!=self.seg[i])*1.0))
            print('self.pts_mask', self.pts_mask.shape, np.sum(self.pts_mask))
            print('eeerror', np.sum((self.sp_seg!=self.seg)*1.0))
        

    def __getitem__(self, item):
        if self.partition == 'train':
            pointcloud = self.data[item][:self.num_points]
            label = self.label[item]
            seg = self.sp_seg[item][:self.num_points]
            mask = self.pts_mask[item][:self.num_points]
            splabel = self.SP_list[item][:self.num_points]

            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
            mask = mask[indices]
            splabel = splabel[indices]

            wa_pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            wa_pointcloud = translate_pointcloud(wa_pointcloud)
            wa_pointcloud = jitter_pointcloud(wa_pointcloud, sigma=0.01, clip=0.02)
            wa_pointcloud = random_mirror_pointcloud(wa_pointcloud)
            return pointcloud, wa_pointcloud, seg, label, mask, splabel

        else:
            pointcloud = self.data[item][:self.num_points]
            label = self.label[item]
            seg = self.seg[item][:self.num_points]

            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
            return pointcloud, seg, label

    def __len__(self):
        return self.data.shape[0]


class ShapeNetModelNet(Dataset):
    def __init__(self, num_points, partition='train', model='meta_add_noisy', idx=[], urate=1.0, class_choice=None):
        # self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice
        # self.partseg_colors = load_color_partseg()

        num_ls = np.load('../MetaSSL/data_seg/ShapeNet/100labeled_idx.npy')
        num_us = np.load('../MetaSSL/data_seg/ShapeNet/100ulabeled_idx.npy')
        # shapes_pick_list = np.load('shapes_pick_list.npy')
        # num_ls = shapes_pick_list[:100]
        # num_us = shapes_pick_list[100:]

        if self.partition == 'train':
            self.data, self.label, self.seg = load_shapenet_train()
            self.data, self.label, self.seg = self.data[num_ls], self.label[num_ls], self.seg[num_ls]
            if len(idx)>0:
                self.data, self.label, self.seg = self.data[idx], self.label[idx], self.seg[idx]
        elif self.partition == 'utrain':

            self.data, self.label, self.seg = load_shapenet_train()
            self.U_data, self.U_label, self.U_seg = self.data[num_us], self.label[num_us], self.seg[num_us]

            self.add_data, self.add_label = load_data_cls('train')


            num_udata = len(self.U_data)
            uall_idx = np.arange(num_udata)
            np.random.shuffle(uall_idx)
            self.U_data = self.U_data[uall_idx[:int(num_udata*urate)]]

            num_udata = len(self.add_data)
            uall_idx = np.arange(num_udata)
            np.random.shuffle(uall_idx)
            self.add_data = self.add_data[uall_idx[:int(num_udata*urate)]]


            self.data = np.concatenate((self.U_data, self.add_data), axis=0)
            print('self.U_data', self.data.shape)

        elif self.partition == 'val':
            self.data, self.label, self.seg = load_shapenet_val()
        elif self.partition == 'test':
            self.data, self.label, self.seg = load_shapenet_test()
        print('self.data', self.data.shape)
        print('self.label', self.label.shape, np.unique(self.label), len(np.unique(self.label)))
        print('self.seg', self.seg.shape, np.unique(self.seg), len(np.unique(self.seg)))

        self.seg_num_all = 50
        self.seg_start_index = 0
        
      
    def __getitem__(self, item):
        if self.partition == 'utrain':
            pointcloud = self.data[item][:self.num_points]
            wa_pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            wa_pointcloud = translate_pointcloud(wa_pointcloud)
            
            np.random.shuffle(pointcloud)
            sa_pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            sa_pointcloud = translate_pointcloud(sa_pointcloud)
            sa_pointcloud = jitter_pointcloud(sa_pointcloud, sigma=0.01, clip=0.02)
            sa_pointcloud = random_mirror_pointcloud(sa_pointcloud)
            opt_sign = np.random.rand()
            if opt_sign>0.3:
                sa_pointcloud = cutout_pointcloud(sa_pointcloud, L=0.5)
            opt_sign = np.random.rand()
            if opt_sign>0.3:
                sa_pointcloud = cut_one_axis(sa_pointcloud)
            return wa_pointcloud, sa_pointcloud, item
        else:
            pointcloud = self.data[item][:self.num_points]
            label = self.label[item]
            seg = self.seg[item][:self.num_points]
            if self.partition == 'train':
                # pointcloud = translate_pointcloud(pointcloud)
                indices = list(range(pointcloud.shape[0]))
                np.random.shuffle(indices)
                pointcloud = pointcloud[indices]
                seg = seg[indices]

            return pointcloud, seg, label

    def __len__(self):
        return self.data.shape[0]



class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition    
        self.semseg_colors = load_color_semseg()

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    data, label = train[0]
    print(data.shape)
    print(label.shape)

    trainval = ShapeNetPart(2048, 'trainval')
    test = ShapeNetPart(2048, 'test')
    data, label, seg = trainval[0]
    print(data.shape)
    print(label.shape)
    print(seg.shape)

    train = S3DIS(4096)
    test = S3DIS(4096, 'test')
    data, seg = train[0]
    print(data.shape)
    print(seg.shape)
