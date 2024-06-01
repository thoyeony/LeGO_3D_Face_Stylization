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

from staticdata import rawlandmarks
import pickle5
import math 
from MICA.mica import MICA 
from preprocessing import remove_eyeball

'''
class VerticesMulti(Dataset):
    def __init__(self, V_ref, F, num_samples=None, num_train = None, normalization = None, mica_model = None, device = None):
        #This class adapted from SIREN https://vsitzmann.github.io/siren/
        super().__init__()
        self.V_ref = V_ref.float().to(device)
        self.all_instances = [
            Vertices(F=F, V_ref=V_ref, num_samples=num_samples, normalization = normalization, mica_model = mica_model, device = device, index = i)
            for i in range(num_train)]
        
        self.num_instances = len(self.all_instances)
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]
        
    def __len__(self):
        return np.sum(self.num_per_instance_observations)
        


    
    def __getitem__(self, idx):
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        
        observations = []
        observations.append(self.all_instances[idx][0])


        ground_truth = [{
            'coords': obj['coords'] ,
            'positions': obj['positions'] ,
            'flame_dict': obj['flame_dict']
            } for obj in observations]

        return observations, ground_truth

class Vertices(Dataset):
    def __init__(self, V_ref, F, num_samples=None, landmarks=None, normalization = None, mica_model = None,  device = None,  index = None):
        super().__init__()
        


        self.device = device

        # Create Flame face mesh 
        torch.manual_seed(index)
        dtype = torch.float32
        expression_codes = torch.tensor(torch.rand((100))*4-torch.ones(100)*2, dtype=dtype, requires_grad=False, device=device).clone().unsqueeze(0)
        shape_codes = torch.tensor(torch.rand((300))*4-torch.ones(300)*2, dtype=dtype, requires_grad=False, device=device).clone().unsqueeze(0)
        pose_value = torch.tensor(round(math.pi/50, 3), dtype=dtype, requires_grad=False, device=device).clone().unsqueeze(0).unsqueeze(0)
        flame_dict = {}
        flame_dict['shape'] = shape_codes
        flame_dict['expression'] = expression_codes
        flame_dict['pose_value'] = pose_value
        self.flame_dict = flame_dict
        mesh_flame = mica_model.get_mesh(flame_dict)   
        mesh_flame= remove_eyeball(input_mesh = (mesh_flame, None)) #Pool out eyes
        mesh_flame = mesh_flame.clone()
        V= normalization.normalize(mesh_flame, F).squeeze(0)#Normalization 
        
        mesh = trimesh.Trimesh(V.squeeze(0).cpu().numpy(), F.cpu().numpy(), process=False)
        mesh_ref = trimesh.Trimesh(V_ref.cpu().numpy(), F.cpu().numpy(), process=False)


        if num_samples > 0:
            V_sample, fids  = trimesh.sample.sample_surface_even(mesh, num_samples)#return {randomly sampled points,corresponding face_index}
            bc = trimesh.triangles.points_to_barycentric(mesh.triangles[fids], V_sample)#return {Barycentric coordinate coefficient}
            V_ref_sample = trimesh.triangles.barycentric_to_points(mesh_ref.triangles[fids], bc)#Randomly select points from trangle 
            self.coords = torch.cat([V_ref, torch.from_numpy(V_ref_sample).to(device)], axis=0)
            self.positions = torch.cat([V, torch.from_numpy(V_sample).to(device)], axis=0)


    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            'coords': self.coords.float().to(self.device),
            'positions': self.positions.float().to(self.device),
            'flame_dict': self.flame_dict
            }


class Sampling(Dataset):
    def __init__(self, V_ref, F, num_samples=3941, device = None):
        super().__init__()
        

        self.V_ref = V_ref
        self.F = F
        self.device = device
        self.num_samples = num_samples
    def sample_points(self, V, s):
        

        # surface points
        positions_list = []
        coords_list = []
        for mesh in V: 
            mesh_ = trimesh.Trimesh(mesh, self.F.clone().cpu().numpy(), process=False)
            mesh_ref = trimesh.Trimesh(self.V_ref.clone().cpu().numpy(), self.F.cpu().numpy(), process=False)
   
         
            V_sample, fids  = trimesh.sample.sample_surface_even(mesh_, self.num_samples)#return {randomly sampled points,corresponding face_index}
            bc = trimesh.triangles.points_to_barycentric(mesh_.triangles[fids], V_sample)#return {Barycentric coordinate coefficient}
            V_ref_sample = trimesh.triangles.barycentric_to_points(mesh_ref.triangles[fids], bc)#Randomly select points from trangle 
            
            
            self.coords = torch.tensor(np.concatenate([self.V_ref.clone().cpu().numpy(), V_ref_sample], axis=0)).float().to(self.device).expand(1,7872,-1)
            self.positions = torch.tensor(np.concatenate([mesh, V_sample], axis=0)).float().to(self.device).expand(1, 7872, -1)
            positions_list.append(self.positions)
            coords_list.append(self.coords)
            
        return {
            'coords': torch.cat(coords_list, dim = 0).to(self.device),
            'positions': torch.cat(positions_list, dim = 0).to(self.device)
            }
    
       
'''      
        
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
        '''
        #save normalized obj file 
        head = trimesh.Trimesh(V, F)
        head.export(os.path.join(self.save_root, name))
        '''
        if name is not None:
            return V, F
        else:
            return V


'''
class TrainData(Dataset):
    def __init__(self, num_train=None, num_samples=3941, use_landmarks=False, device = None):
        
        normalization = Normalize()
        
        #Normalize template head mesh 
        template_head_path = './staticdata/eyeball_removed_head_template.obj'
        template_head_mesh = trimesh.load(template_head_path)
        template_name = template_head_path.split('/')[-1]
        template_vertices, template_faces = normalization.normalize(torch.tensor(template_head_mesh.vertices).to(device), 
                                                                    torch.tensor(template_head_mesh.faces).to(device), template_name)
        
        mica_checkpoint = './mica.tar'
        mica_model = MICA(mica_checkpoint, device)
        
        self.vertices = torch.tensor(template_vertices).clone().float().to(device)
        self.faces = torch.tensor(template_faces).clone().to(device)
                
        self.data = VerticesMulti(self.vertices, self.faces, num_samples=num_samples, 
                                num_train = num_train, normalization = normalization, mica_model = mica_model, device = device)
        
        
    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend( [bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                   ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)
'''
class TrainData(Dataset):
    def __init__(self, num_train=None, num_samples=3941, use_landmarks=False, device = None):
        
        normalization = Normalize()
        
        #Normalize template head mesh 
        template_head_path = './staticdata/eyeball_removed_head_template.obj'
        template_head_mesh = trimesh.load(template_head_path)
        template_name = template_head_path.split('/')[-1]
        template_vertices, template_faces = normalization.normalize(torch.tensor(template_head_mesh.vertices).to(device), 
                                                                    torch.tensor(template_head_mesh.faces).to(device), template_name)
        
        mica_checkpoint = './mica.tar'
        mica_model = MICA(mica_checkpoint, device)
        
        self.vertices = torch.tensor(template_vertices).clone().float().to(device)
        self.faces = torch.tensor(template_faces).clone().to(device)
                