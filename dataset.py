import os
import random
import pickle

import torch
import trimesh
import igl

import numpy as np
from glob import glob
from cyobj.io import read_obj
from torch.utils.data import Dataset
from scipy.io import loadmat
import skimage.io as sio
import skimage.filters as sfi

import pickle5
import math 
from MICA.mica import MICA 

    
        
class Normalize(Dataset):
    def __init__(self, dir= None, num_train=None):
        super().__init__()
        self.dataset_root = dir
        self.num_train = num_train
        #self.list_of_meshes = os.listdir(self.dataset_root)
        #self.scale_mean_shape()
    def compute_mean_shape(self):
        vertices_sum = np.zeros((5023, 3))
        
        for idx in range(self.num_train):
            mesh = trimesh.load(os.path.join(self.dataset_root, self.list_of_meshes[idx]))
            vertices_sum += mesh.vertices
        mean_shape = vertices_sum / self.num_train
        return mean_shape
    
    def center_mean_shape(self):
        mean_shape = self.compute_mean_shape()
        self.mean_center = np.mean(mean_shape, axis = 0)
        mean_shape -= self.mean_center
        return mean_shape
    
    def scale_mean_shape(self):
        mean_shape = self.center_mean_shape()
        self.max_distance = np.max(np.linalg.norm(mean_shape, axis = 1))
        mean_shape /= self.max_distance
        self.mean_shape = mean_shape
    
    def normalize(self, V, F = None, name= None):
        device = V.device
        self.mean_center = torch.tensor([ 1.99295308e-04, -4.31300949e-01, -2.05403929e-01]).to(device)
        self.max_distance = 228.0089383671405
        V-=self.mean_center
        V/=self.max_distance*0.5
      
        if name is not None:
            return V, F
        else:
            return V



class TrainData(Dataset):
    def __init__(self, device = None):
        
        normalization = Normalize()
        
        #Normalize template head mesh 
        template_head_path = './staticdata/template_FLAME.obj'
        template_head_mesh = trimesh.load(template_head_path)
        template_name = template_head_path.split('/')[-1]
        template_vertices, template_faces = normalization.normalize(torch.tensor(template_head_mesh.vertices).to(device), 
                                                                    torch.tensor(template_head_mesh.faces).to(device), template_name)
        
        mica_checkpoint = './mica.tar'
        mica_model = MICA(mica_checkpoint, device)
        
        self.vertices = torch.tensor(template_vertices).clone().float().to(device)
        self.faces = torch.tensor(template_faces).clone().to(device)
                