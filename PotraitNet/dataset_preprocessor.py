import torch
import torch.utils.data as data
import numpy as np

import os
import cv2
import random
import copy
from PIL import Image

from data_aug import data_aug_blur, data_aug_color, data_aug_noise, data_aug_light
from data_aug import data_aug_flip, flip_data, aug_matrix
from data_aug import show_edge, mask_to_bbox, load_json
from data_aug import base64_2_mask, mask_2_base64, padding, Normalize_Img, Anti_Normalize_Img


class Human(data.Dataset): 
    def __init__(self, exp_args):
        self.exp_args = exp_args
        
        self.datasets = {}
        self.imagelist = []

        ImageRoot = 'data\EG1800\Images\\'
        AnnoRoot = 'data\EG1800\Labels\\'
        ImgIds_Train = 'data\eg1800_train.txt'
        ImgIds_Test = 'data\eg1800_test.txt'
        exp_args.dataset = 'eg1800'
        self.datasets['eg1800'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)

        for key in self.datasets.keys():
            length = len(self.datasets[key])
            for i in range(length):
                self.imagelist.append([key, i])
        
    def __getitem__(self, index):
        subset, subsetidx = self.imagelist[index]
        
        #if self.task == 'seg':
        input_ori, input, output_edge, output_mask = self.datasets[subset][subsetidx]
        return input_ori.astype(np.float32), input.astype(np.float32), output_edge.astype(np.int64), output_mask.astype(np.int64)
           
    def __len__(self):
        return len(self.imagelist)
    


class PortraitSeg(data.Dataset): 
    def __init__(self, ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, exp_args):
        self.ImageRoot = ImageRoot
        self.AnnoRoot = AnnoRoot
        self.istrain = exp_args.istrain
                
        #self.task = exp_args.task
        self.dataset = exp_args.dataset
        self.input_height = exp_args.input_height
        self.input_width = exp_args.input_width
        
        self.padding_color = exp_args.padding_color
        self.img_scale = exp_args.img_scale
        self.img_mean = exp_args.img_mean # BGR order
        self.img_val = exp_args.img_val # BGR order
        
        if self.istrain == True:
            file_object = open(ImgIds_Train, 'r')
        elif self.istrain == False:
            file_object = open(ImgIds_Test, 'r')
            
        try:
            self.imgIds = file_object.readlines()                
        finally:
             file_object.close()
        pass
            
        
    def __getitem__(self, index):
        '''
        An item is an image. Which may contains more than one person.
        '''
        img, mask, bbox, H = None, None, None, None
            
        # basic info
        img_id = self.imgIds[index].strip()
        img_path = os.path.join(self.ImageRoot, img_id)
        img = cv2.imread(img_path)
        
        # load mask
        annopath = os.path.join(self.AnnoRoot, img_id.replace('.jpg', '.png'))
        mask = cv2.imread(annopath, 0)
        mask[mask>1] = 0
        
        height, width, _ = img.shape
        bbox = [0, 0, width-1, height-1]
        H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                    angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height/4)
        
        use_float_mask = False # use original 0/1 mask as groundtruth
        
        # data augument: first align center to center of dst size. then rotate and scale
        if self.istrain == False:
            img_aug_ori, mask_aug_ori = padding(img, mask, size=self.input_width, padding_color=self.padding_color)
            
            # ===========add new channel for video stability============
            input_norm = Normalize_Img(img_aug_ori, scale=self.img_scale, mean=self.img_mean, val=self.img_val)
            input = np.transpose(input_norm, (2, 0, 1))
            input_ori = copy.deepcopy(input)
        else:
            img_aug = cv2.warpAffine(np.uint8(img), H, (self.input_width, self.input_height), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(self.padding_color, self.padding_color, self.padding_color)) 
            mask_aug = cv2.warpAffine(np.uint8(mask), H, (self.input_width, self.input_height), 
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            img_aug_ori, mask_aug_ori, aug_flag = data_aug_flip(img_aug, mask_aug)
            prior = np.zeros((self.input_height, self.input_width, 1))
                
            # add augmentation
            img_aug = Image.fromarray(cv2.cvtColor(img_aug_ori, cv2.COLOR_BGR2RGB))  
            img_aug = data_aug_color(img_aug)
            img_aug = np.asarray(img_aug)
            img_aug = data_aug_blur(img_aug)
            img_aug = data_aug_noise(img_aug)
            img_aug = np.float32(img_aug[:,:,::-1]) # BGR, like cv2.imread
            
            input_norm = Normalize_Img(img_aug, scale=self.img_scale, mean=self.img_mean, val=self.img_val)
            input_ori_norm = Normalize_Img(img_aug_ori, scale=self.img_scale, mean=self.img_mean, val=self.img_val)
                
            input = np.transpose(input_norm, (2, 0, 1))
            input_ori = np.transpose(input_ori_norm, (2, 0, 1))
            
        #if 'seg' in self.task:
        # if use_float_mask == True:
        #     output_mask = cv2.resize(mask_aug_ori, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
        #     cv2.normalize(output_mask, output_mask, 0, 1, cv2.NORM_MINMAX)
        #     output_mask[output_mask>=0.5] = 1
        #     output_mask[output_mask<0.5] = 0
        # else:
        output_mask = cv2.resize(np.uint8(mask_aug_ori), (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
        
        # add mask blur
        output_mask = np.uint8(cv2.blur(output_mask, (5,5)))
        output_mask[output_mask>=0.5] = 1
        output_mask[output_mask<0.5] = 0
        #else:
        #    output_mask = np.zeros((self.input_height, self.input_width), dtype=np.uint8) + 255
        
        #if self.task == 'seg':
        edge = show_edge(output_mask)
        # edge_blur = np.uint8(cv2.blur(edge, (5,5)))/255.0
        return input_ori, input, edge, output_mask
            
    def __len__(self):
        return len(self.imgIds)